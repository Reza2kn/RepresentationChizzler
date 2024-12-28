import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from typing import Tuple, List
from dotenv import load_dotenv
import os
import gradio as gr
import tempfile
import json

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# Initialize the model globally to avoid reloading
print("Initializing Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True,
    trust_repo=True
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model loaded successfully (using {device})")

def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio file and convert to required format."""
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    return waveform, sample_rate

def get_speech_timestamps(
    vad_model,
    waveform: torch.Tensor,
    sample_rate: int,
    utils,
    threshold: float = 0.5,
    min_silence_duration: float = 4.0
) -> List[dict]:
    """Get speech timestamps using Silero VAD."""
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    
    speech_timestamps = get_speech_timestamps(
        waveform,
        vad_model,
        threshold=threshold,
        return_seconds=True
    )
    
    return speech_timestamps

def merge_close_segments(segments: List[dict], max_gap: float = 4.0) -> List[dict]:
    """Merge speech segments that are close together."""
    if not segments:
        return segments
    
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
    return merged

def process_audio(audio_path: str, threshold: float = 0.5, max_gap: float = 4.0) -> Tuple[str, str]:
    """Process a single audio file and return the cleaned version and processing details."""
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        output_path = temp_file.name
    
    # Process audio
    waveform, sample_rate = load_audio(audio_path)
    speech_timestamps = get_speech_timestamps(model, waveform, sample_rate, utils, threshold)
    merged_timestamps = merge_close_segments(speech_timestamps, max_gap)
    
    # Prepare processing details
    details = []
    audio = AudioSegment.from_mp3(audio_path)
    final_audio = AudioSegment.empty()
    
    if merged_timestamps:
        for i, segment in enumerate(merged_timestamps):
            start_time = segment['start'] * 1000
            end_time = segment['end'] * 1000
            
            segment_audio = audio[start_time:end_time]
            final_audio += segment_audio
            
            details.append(
                f"Segment {i+1}/{len(merged_timestamps)}: "
                f"{segment['start']:.2f}s to {segment['end']:.2f}s "
                f"(duration: {(end_time-start_time)/1000:.2f}s)"
            )
        
        final_audio.export(output_path, format="mp3")
        details.append(f"\nFinal duration: {len(final_audio)/1000:.2f}s")
    else:
        details.append("No speech detected in the audio file!")
        return audio_path, "\n".join(details)
    
    return output_path, "\n".join(details)

def gradio_interface(audio_file, vad_threshold, max_silence_gap):
    """Gradio interface function"""
    if audio_file is None:
        return None, "Please upload an audio file."
    
    cleaned_audio, details = process_audio(
        audio_file,
        threshold=vad_threshold,
        max_gap=max_silence_gap
    )
    
    return cleaned_audio, details

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(label="Upload Audio File", type="filepath"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.1, 
                 label="VAD Threshold (higher = stricter voice detection)"),
        gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                 label="Max Silence Gap (seconds)")
    ],
    outputs=[
        gr.Audio(label="Cleaned Audio"),
        gr.Textbox(label="Processing Details", lines=10)
    ],
    title="Representation Chizzler™🔥",
    description="""Upload an audio file to remove non-speech segments and clean it up.
    The tool will detect speech segments and remove silence and noise between them.
    Adjust the VAD threshold and maximum silence gap to fine-tune the cleaning process.""",
    examples=[],  # You can add example audio files here
    theme="default"
)

if __name__ == "__main__":
    iface.launch(share=True)
    