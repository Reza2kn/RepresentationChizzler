# 🎧 Representation Chizzler™

A powerful two-stage audio processing tool that combines Voice Activity Detection (VAD) and Speech Enhancement to clean and denoise audio files.

## 🌟 Features

1. **Two-Stage Processing Pipeline**:
   - Stage 1: Uses Silero VAD to detect and extract speech segments
   - Stage 2: Applies MP-SENet deep learning model to remove noise

2. **Memory-Efficient Processing**:
   - Processes audio in chunks to prevent memory issues
   - Automatically converts audio to the required format (16kHz mono WAV)

3. **User-Friendly Interface**:
   - Beautiful Gradio web interface
   - Real-time progress reporting
   - Compare original, VAD-processed, and denoised versions

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Reza2kn/RepresentationChizzler.git
   cd RepresentationChizzler
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Hugging Face token:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```

5. Download MP-SENet:
   - Clone the MP-SENet repository:
     ```bash
     git clone https://github.com/yxlu-0102/MP-SENet.git
     ```
   - Download the model checkpoint and config files:
     - Place `g_best_dns` in `MP-SENet/best_ckpt/`
     - Place `config.json` in `MP-SENet/best_ckpt/`

## 🎮 Usage

1. Run the app:
   ```bash
   python run.py
   ```

2. Open your web browser and navigate to the provided URL

3. Upload an audio file and adjust the parameters:
   - VAD Threshold: Controls voice detection sensitivity (0.1-0.9)
   - Max Silence Gap: Controls merging of close speech segments (1-10s)

4. Compare the results:
   - Original Audio
   - VAD Processed (Speech Only)
   - Final Denoised

## 🛠️ Parameters

- **VAD Threshold** (0.1-0.9):
  - Higher values = stricter voice detection
  - Lower values = more lenient detection
  - Default: 0.5

- **Max Silence Gap** (1-10s):
  - Maximum silence duration to consider segments as continuous
  - Higher values = fewer segments but may include more silence
  - Default: 4.0s

## 🙏 Credits

This project combines two powerful models:
- [Silero VAD](https://github.com/snakers4/silero-vad) for Voice Activity Detection
- [MP-SENet](https://github.com/yxlu-0102/MP-SENet) for Speech Enhancement

## 📝 License

This project is licensed under the terms specified in the MP-SENet repository.
