import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp as youtube_dl
from huey import RedisHuey
import torch.multiprocessing as mp

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Configure Huey instance
huey = RedisHuey('huey_gary', host='127.0.0.1', port=6379)

app = Flask(__name__)
CORS(app)

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

def load_and_preprocess_audio(file_path):
    song, sr = torchaudio.load(file_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    song = song.to(device)
    expected_sr = 32000
    if sr != expected_sr:
        resampler = T.Resample(sr, expected_sr).to(device)
        song = resampler(song)
        sr = expected_sr
    prompt_length = sr * 5
    prompt_waveform = song[:, :prompt_length] if song.shape[1] > prompt_length else song
    return prompt_waveform, sr

def generate_audio_continuation(prompt_waveform, sr):
    model_continue = MusicGen.get_pretrained('facebook/musicgen-small')
    model_continue.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=12, cfg_coef=3)
    output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)
    return output.cpu().squeeze(0)

def save_generated_audio(output, sr):
    output_filename = 'generated_continuation'
    audio_write(output_filename, output, sr, strategy="loudness", loudness_compressor=True)
    return output_filename + '.wav'

@huey.task()
def process_youtube_url(youtube_url):
    try:
        # Ensure CUDA operations are initialized in the task function
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(device)

        downloaded_mp3, downloaded_webm = download_audio(youtube_url)
        prompt_waveform, sr = load_and_preprocess_audio(downloaded_mp3)

        # Load the model inside the task function
        model_continue = MusicGen.get_pretrained('facebook/musicgen-small')
        model_continue.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=12, cfg_coef=3)

        output = generate_audio_continuation(prompt_waveform, sr)
        output_filename = save_generated_audio(output, sr)

        cleanup_files(downloaded_mp3, downloaded_webm)
        return output_filename
    except Exception as e:
        print(f"Error processing YouTube URL: {e}")
        return None

# Flask route to handle task creation
@app.route('/generate', methods=['POST'])
def generate_audio():
    data = request.json
    youtube_url = data['url']

    # Enqueue the task
    task = process_youtube_url(youtube_url)

    # Return the task ID to the client
    return jsonify({"task_id": task.id})

if __name__ == '__main__':
    app.run(debug=True)
