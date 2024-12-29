import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from typing import Tuple, List, Dict
from dotenv import load_dotenv
import os
import gradio as gr
import tempfile
import json
import sys
from pathlib import Path
import soundfile as sf
from datetime import datetime
from rich.console import Console
import time

# Add MP-SENet to path
sys.path.append(str(Path(__file__).parent / "MP-SENet"))
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
from env import AttrDict

# Initialize console for pretty printing
console = Console()

def log_progress(message: str, level: int = 1) -> None:
    """Log a progress message with timestamp and indentation."""
    indent = "  " * (level - 1)
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{timestamp}[/dim] {indent}[bold blue]►[/bold blue] {message}")
    sys.stdout.flush()  # Force flush stdout separately

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# Initialize models
def initialize_models():
    log_progress("🎯 Initializing models...")
    
    # Use CPU for all operations
    device = torch.device("cpu")
    log_progress("💻 Using CPU for all operations", 2)
    
    # Initialize Silero VAD
    log_progress("🎤 Loading Silero VAD model...", 2)
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        trust_repo=True
    )
    vad_model = model.to(device)
    log_progress("✅ Silero VAD loaded successfully", 2)
    
    # Initialize MP-SENet
    log_progress("🎧 Loading MP-SENet model...", 2)
    checkpoint_path = str(Path(__file__).parent / "MP-SENet/best_ckpt/g_best_dns")
    config_file = str(Path(checkpoint_path).parent / "config.json")
    
    with open(config_file) as f:
        config = AttrDict(json.loads(f.read()))
    
    mpnet_model = MPNet(config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    mpnet_model.load_state_dict(state_dict['generator'])
    mpnet_model.eval()
    log_progress("✅ MP-SENet loaded successfully", 2)
    
    return vad_model, utils, mpnet_model, config, device

# Load models globally
vad_model, vad_utils, mpnet_model, mpnet_config, device = initialize_models()

def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio file and convert to required format."""
    log_progress(f"🎵 Loading audio: {Path(file_path).name}")
    
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        log_progress("Converting stereo to mono...", 2)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != 16000:
        log_progress(f"Resampling from {sample_rate}Hz to 16000Hz...", 2)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    return waveform, sample_rate

def get_speech_timestamps(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.5,
    min_silence_duration: float = 4.0
) -> List[dict]:
    """Get speech timestamps using Silero VAD."""
    log_progress("🔍 Detecting speech segments...")
    
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = vad_utils
    
    speech_timestamps = get_speech_timestamps(
        waveform,
        vad_model,
        threshold=threshold,
        return_seconds=True
    )
    
    log_progress(f"✅ Found {len(speech_timestamps)} speech segments", 2)
    return speech_timestamps

def merge_close_segments(segments: List[dict], max_gap: float = 4.0) -> List[dict]:
    """Merge speech segments that are close together."""
    if not segments:
        return segments
    
    log_progress("🤝 Merging close segments...")
    merged = []
    current_segment = segments[0].copy()
    
    for segment in segments[1:]:
        gap_duration = segment['start'] - current_segment['end']
        
        if gap_duration <= max_gap:
            current_segment['end'] = segment['end']
        else:
            merged.append(current_segment)
            current_segment = segment.copy()
    
    merged.append(current_segment)
    log_progress(f"✅ Merged into {len(merged)} segments", 2)
    return merged

def denoise_audio_chunk(audio_tensor: torch.Tensor, chunk_size: int = 5 * 16000) -> torch.Tensor:
    """Denoise a chunk of audio using MP-SENet."""
    # Process in 5-second chunks
    chunks = []
    for i in range(0, audio_tensor.size(1), chunk_size):
        chunk = audio_tensor[:, i:min(i + chunk_size, audio_tensor.size(1))]
        
        # Normalize
        norm_factor = torch.sqrt(chunk.size(1) / torch.sum(chunk ** 2.0, dim=1))
        chunk = chunk * norm_factor.unsqueeze(1)
        
        # Process through model
        with torch.no_grad():
            noisy_amp, noisy_pha, _ = mag_pha_stft(
                chunk, mpnet_config.n_fft, mpnet_config.hop_size, 
                mpnet_config.win_size, mpnet_config.compress_factor
            )
            amp_g, pha_g, _ = mpnet_model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(
                amp_g, pha_g, mpnet_config.n_fft, mpnet_config.hop_size,
                mpnet_config.win_size, mpnet_config.compress_factor
            )
        
        # Denormalize
        audio_g = audio_g / norm_factor.unsqueeze(1)
        chunks.append(audio_g)
        
        # Clean up
        del chunk, noisy_amp, noisy_pha, amp_g, pha_g
    
    return torch.cat(chunks, dim=1)

def process_audio(
    audio_path: str, 
    threshold: float = 0.5, 
    max_gap: float = 4.0
) -> Tuple[str, str, str, str]:
    """Process audio file in two stages: VAD and denoising."""
    log_progress(f"🚀 Processing: {Path(audio_path).name}")
    
    # Stage 1: VAD Processing
    log_progress("📊 Stage 1: Voice Activity Detection", 2)
    waveform, sample_rate = load_audio(audio_path)
    speech_timestamps = get_speech_timestamps(waveform, sample_rate, threshold)
    merged_timestamps = merge_close_segments(speech_timestamps, max_gap)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as vad_file, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as denoised_file:
        vad_path = vad_file.name
        denoised_path = denoised_file.name
    
    # Extract speech segments
    log_progress("✂️ Extracting speech segments...", 2)
    audio = AudioSegment.from_file(audio_path)
    final_audio = AudioSegment.empty()
    details = ["🎯 Processing Details:"]
    
    if merged_timestamps:
        for i, segment in enumerate(merged_timestamps, 1):
            start_time = segment['start'] * 1000
            end_time = segment['end'] * 1000
            
            segment_audio = audio[start_time:end_time]
            final_audio += segment_audio
            
            details.append(
                f"🎵 Segment {i}/{len(merged_timestamps)}: "
                f"{segment['start']:.1f}s to {segment['end']:.1f}s "
                f"(duration: {(end_time-start_time)/1000:.1f}s)"
            )
        
        # Save VAD-processed audio
        final_audio.export(vad_path, format="wav")
        vad_duration = len(final_audio)/1000
        details.append(f"\n⏱️ VAD Output Duration: {vad_duration:.1f}s")
        details.append(f"📉 Reduced by: {(1 - vad_duration/(len(audio)/1000))*100:.1f}%")
        
        # Stage 2: Denoising
        log_progress("🎧 Stage 2: MP-SENet Denoising", 2)
        try:
            # Load VAD output
            vad_tensor, _ = load_audio(vad_path)
            
            # Process through MP-SENet
            log_progress("🔄 Applying noise reduction...", 2)
            with torch.no_grad():
                denoised_tensor = denoise_audio_chunk(vad_tensor)
            
            # Save denoised audio
            sf.write(denoised_path, denoised_tensor.numpy().squeeze(), mpnet_config.sampling_rate)
            log_progress("✨ Denoising complete!", 2)
            
        except Exception as e:
            log_progress(f"❌ Error during denoising: {str(e)}", 2)
            return audio_path, audio_path, audio_path, "\n".join(details + [f"\n❌ Denoising failed: {str(e)}"])
    else:
        details.append("❌ No speech detected in the audio file!")
        return audio_path, audio_path, audio_path, "\n".join(details)
    
    log_progress("✅ Processing complete!")
    return audio_path, vad_path, denoised_path, "\n".join(details)

def gradio_interface(audio_file, vad_threshold, max_silence_gap):
    """Gradio interface function"""
    if audio_file is None:
        return None, None, None, "❌ Please upload an audio file."
    
    original_path, vad_path, denoised_path, details = process_audio(
        audio_file,
        threshold=vad_threshold,
        max_gap=max_silence_gap
    )
    
    return original_path, vad_path, denoised_path, details

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(label="🎵 Upload Audio File", type="filepath"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.1, 
                 label="🎯 VAD Threshold (higher = stricter voice detection)"),
        gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                 label="⏱️ Max Silence Gap (seconds)")
    ],
    outputs=[
        gr.Audio(label="📝 Original Audio"),
        gr.Audio(label="✂️ VAD Processed (Speech Only)"),
        gr.Audio(label="✨ Final Denoised"),
        gr.Textbox(label="📊 Processing Details", lines=10)
    ],
    title="🔥 Representation Chizzler™ v2.0",
    description="""
    🎯 A two-stage audio processing tool:
    1️⃣ First, it uses Silero VAD to detect and extract speech segments
    2️⃣ Then, it applies MP-SENet deep learning model to remove noise
    
    📝 Instructions:
    - Upload any audio file
    - Adjust the VAD threshold to control voice detection sensitivity
    - Set the maximum silence gap to control segment merging
    - Compare all three versions: original, speech-only, and denoised!
    """,
    theme="default"
)

if __name__ == "__main__":
    iface.launch(share=True)