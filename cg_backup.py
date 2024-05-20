import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp as youtube_dl
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio.transforms as T
from concurrent.futures import ThreadPoolExecutor
from flask import current_app as app
import json

app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor(max_workers=2)

def cleanup_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def download_audio(youtube_url):
    downloaded_mp3 = 'downloaded_audio.mp3'
    downloaded_webm = 'downloaded_audio.webm'
    cleanup_files(downloaded_mp3, downloaded_webm)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'keepvideo': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return downloaded_mp3, downloaded_webm

def load_and_preprocess_audio(file_path, timestamp):
   song, sr = torchaudio.load(file_path)
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   song = song.to(device)
   expected_sr = 32000
   if sr != expected_sr:
       resampler = T.Resample(sr, expected_sr).to(device)
       song = resampler(song)
       sr = expected_sr

   # Convert timestamp (seconds) to frames
   frame_offset = int(timestamp * sr)

   # Check if waveform duration after timestamp is less than 30 seconds
   if song.shape[1] - frame_offset < 30 * sr:
       # Wrap around to the beginning of the mp3
       song = torch.cat((song[:, frame_offset:], song[:, :30 * sr - (song.shape[1] - frame_offset)]), dim=1)
   else:
       song = song[:, frame_offset:frame_offset + 30 * sr]

   # Define the prompt length
   prompt_length = 6 * sr

   # Create the prompt waveform
   prompt_waveform = song[:, :prompt_length] if song.shape[1] > prompt_length else song

   return prompt_waveform, sr

def generate_audio_continuation(prompt_waveform, sr):
    model_continue = MusicGen.get_pretrained('facebook/musicgen-small')
    model_continue.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=16, cfg_coef=3)
    output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)
    return output.cpu().squeeze(0)

def save_generated_audio(output, sr):
    output_filename = 'generated_continuation'
    audio_write(output_filename, output, sr, strategy="loudness", loudness_compressor=True)
    return output_filename + '.wav'

def process_youtube_url(youtube_url, timestamp):
    try:
        downloaded_mp3, downloaded_webm = download_audio(youtube_url)
        prompt_waveform, sr = load_and_preprocess_audio(downloaded_mp3, timestamp)
        output = generate_audio_continuation(prompt_waveform, sr)
        output_filename = save_generated_audio(output, sr)
        cleanup_files(downloaded_mp3, downloaded_webm)
        return output_filename
    except Exception as e:
        print(f"Error processing YouTube URL: {e}")
        return None

@app.route('/generate', methods=['POST'])
def generate_audio():
    data = request.json
    youtube_url = data['url']
    print_data = request.get_json()
    pretty_data = json.dumps(print_data, indent=4)  # Pretty print the JSON data
    app.logger.info(f'JSON data received: \n{pretty_data}')  # Log the entire JSON data
    timestamp = data.get('currentTime')  # Get the timestamp, default to 0 if not provided

    # Log the timestamp
    app.logger.info(f'Timestamp received: {timestamp}')

    audio_path = process_youtube_url(youtube_url, timestamp)
    if audio_path:
        with open(audio_path, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        cleanup_files(audio_path)
        return jsonify({"audio": encoded_audio})
    else:
        return jsonify({"error": "Failed to process audio"}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)