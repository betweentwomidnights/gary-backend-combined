import os
import base64
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import yt_dlp as youtube_dl
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio.transforms as T
from concurrent.futures import ThreadPoolExecutor
import json
import librosa
import soundfile as sf

from rq import Queue, Retry
from redis import Redis

from pymongo import MongoClient, errors

from bson import ObjectId, json_util
import bson  # Import bson to handle bson-related errors
import re

from g4laudio import continue_music
import gc
import time

# MongoDB connection with retry logic
def get_mongo_client():
    try:
        client = MongoClient('mongodb://mongo:27017/', serverSelectionTimeoutMS=60000)
        client.admin.command('ping')  # Check if the connection is established
        return client
    except errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        return None

client = get_mongo_client()
if client:
    db = client['name']
    audio_tasks = db.audio_tasks
else:
    print("Failed to connect to MongoDB.")

# Redis connection
redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
print(f"Connecting to Redis at '{redis_url}'")
redis_conn = Redis.from_url(redis_url)
q = Queue(connection=redis_conn)

app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor(max_workers=24)

def is_valid_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?\s]{11})')
    youtube_pattern = re.compile(youtube_regex)
    return re.match(youtube_pattern, url) is not None

def cleanup_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path) and file_path.endswith('.webm'):
            os.remove(file_path)

def download_audio(youtube_url):
    cache_dir = '/dataset/gary'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Check Redis cache
    audio_id = base64.urlsafe_b64encode(youtube_url.encode()).decode('utf-8')
    cached_mp3_path = redis_conn.get(audio_id)

    if cached_mp3_path:
        cached_mp3_path = cached_mp3_path.decode('utf-8')
        if os.path.exists(cached_mp3_path):
            print(f"Using cached audio for URL: {youtube_url}")
            return cached_mp3_path

    downloaded_mp3 = 'downloaded_audio.mp3'
    downloaded_webm = 'downloaded_audio.webm'
    cleanup_files(downloaded_webm)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'keepvideo': False,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Move the downloaded file to the cache directory
    cached_mp3_path = os.path.join(cache_dir, f'{audio_id}.mp3')
    os.rename(downloaded_mp3, cached_mp3_path)
    cleanup_files(downloaded_webm)

    # Store the cached file path in Redis
    redis_conn.set(audio_id, cached_mp3_path)

    return cached_mp3_path

def get_bpm(cached_mp3_path):
    audio, sr = librosa.load(cached_mp3_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if 120 < tempo < 200:
        tempo = tempo / 2
    return tempo

def calculate_duration(bpm, min_duration, max_duration):
    single_bar_duration = 4 * 60 / bpm
    bars = max(min_duration // single_bar_duration, 1)

    while single_bar_duration * bars < min_duration:
        bars += 1

    duration = single_bar_duration * bars

    while duration > max_duration and bars > 1:
        bars -= 1
        duration = single_bar_duration * bars

    return duration

def load_and_preprocess_audio(file_path, timestamp, promptLength):
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
    prompt_length = promptLength * sr

    # Create the prompt waveform
    prompt_waveform = song[:, :prompt_length] if song.shape[1] > prompt_length else song

    return prompt_waveform, sr

def generate_audio_continuation(prompt_waveform, sr, bpm, model, min_duration, max_duration, progress_callback=None):
    # Calculate the duration to end at a bar
    duration = calculate_duration(bpm, min_duration, max_duration)

    # Use a new CUDA stream for this task
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model_continue = MusicGen.get_pretrained(model)
        model_continue.set_custom_progress_callback(progress_callback)
        model_continue.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=duration, cfg_coef=3)
        output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)
    return output.cpu().squeeze(0)

def save_generated_audio(output, sr):
    output_filename = 'generated_continuation'
    audio_write(output_filename, output, sr, strategy="loudness", loudness_compressor=True)
    return output_filename + '.wav'

def process_youtube_url(youtube_url, timestamp, model, promptLength, min_duration, max_duration, task_id):
    try:
        def progress_callback(current_step, total_steps):
            progress_percent = (current_step / total_steps) * 100
            print(f"Progress: {progress_percent}% for task {task_id}")
            redis_conn.set(f"progress_{task_id}", progress_percent, ex=600)  # Set progress with a TTL of 600 seconds

        cached_mp3_path = download_audio(youtube_url)
        bpm = get_bpm(cached_mp3_path)
        prompt_waveform, sr = load_and_preprocess_audio(cached_mp3_path, timestamp, promptLength)
        output = generate_audio_continuation(prompt_waveform, sr, bpm, model, min_duration, max_duration, progress_callback)
        output_filename = save_generated_audio(output, sr)

        # Encode the audio data
        with open(output_filename, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        # Save task info, audio reference, and status in MongoDB
        audio_tasks.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'output_filename': output_filename, 'status': 'completed', 'audio': encoded_audio}}
        )

        return output_filename
    except Exception as e:
        print(f"Error processing YouTube URL: {e}")
        # Update the task status in MongoDB in case of an error
        audio_tasks.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'status': 'failed'}}
        )
        return None

def process_continuation(task_id, input_data_base64, musicgen_model, prompt_duration):
    try:
        def progress_callback(current_step, total_steps):
            progress_percent = (current_step / total_steps) * 100
            print(f"Progress: {progress_percent}% for task {task_id}")
            redis_conn.set(f"progress_{task_id}", progress_percent, ex=600)  # Set progress with a TTL of 600 seconds

        print(f"Memory before DB find: {torch.cuda.memory_allocated()} bytes")
        task = audio_tasks.find_one({'_id': ObjectId(task_id)})
        print(f"Memory after DB find: {torch.cuda.memory_allocated()} bytes")
        if not task:
            print("Task not found")
            return None

        output_data_base64 = continue_music(input_data_base64, musicgen_model, progress_callback=progress_callback, prompt_duration=prompt_duration)
        task['audio'] = output_data_base64
        task['status'] = 'completed'

        print(f"Memory before DB update: {torch.cuda.memory_allocated()} bytes")
        audio_tasks.update_one({'_id': ObjectId(task_id)}, {"$set": task})
        print(f"Memory after DB update: {torch.cuda.memory_allocated()} bytes")

        return output_data_base64
    except Exception as e:
        print(f"Error processing continuation: {e}")
        # Update the task status in MongoDB in case of an error
        audio_tasks.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'status': 'failed'}}
        )
        return None

@app.route('/generate', methods=['POST'])
def generate_audio():
    data = request.json
    youtube_url = data['url']
    timestamp = data.get('currentTime', 0)
    model = data.get('model', 'facebook/musicgen-small')
    promptLength = int(data.get('promptLength', 6))
    duration = data.get('duration', '16-18').split('-')

    # Ensure that duration is correctly parsed and handled
    min_duration = int(duration[0])
    max_duration = int(duration[1])

    # Validate YouTube URL
    if not is_valid_youtube_url(youtube_url):
        return jsonify({"error": "Invalid YouTube URL"}), 400

    # Validate timestamp
    if not isinstance(timestamp, (int, float)) or timestamp < 0:
        return jsonify({"error": "Invalid timestamp"}), 400

    # Save task info in MongoDB
    audio_task = {
        'rq_job_id': None,
        'youtube_url': youtube_url,
        'timestamp': timestamp,
        'status': 'pending'
    }
    task_id = audio_tasks.insert_one(audio_task).inserted_id

    # Enqueue the task with retry logic
    job = q.enqueue(
        process_youtube_url,
        youtube_url,
        timestamp,
        model,
        promptLength,
        min_duration,
        max_duration,
        str(task_id),
        job_timeout=600,
        retry=Retry(max=3)
    )

    # Update the job ID in the MongoDB task record
    audio_tasks.update_one({'_id': ObjectId(task_id)}, {'$set': {'rq_job_id': job.get_id()}})

    return jsonify({"task_id": str(task_id)})

@app.route('/continue', methods=['POST'])
def continue_audio():
    data = request.json
    task_id = data['task_id']
    musicgen_model = data['model']
    prompt_duration = int(data.get('prompt_duration', 6))
    input_data_base64 = data['audio']  # Get the audio data from the request

    # Validate task ID
    if not ObjectId.is_valid(task_id):
        return jsonify({"error": "Invalid task ID"}), 400

    # Save task info in MongoDB
    audio_task = audio_tasks.find_one({'_id': ObjectId(task_id)})
    if not audio_task:
        return jsonify({"error": "Task not found"}), 404

    # Enqueue the task with retry logic
    job = q.enqueue(
        process_continuation,
        str(task_id),
        input_data_base64,
        musicgen_model,
        prompt_duration,
        job_timeout=600,
        retry=Retry(max=3)
    )

    # Update the job ID in the MongoDB task record
    audio_tasks.update_one({'_id': ObjectId(task_id)}, {'$set': {'rq_job_id': job.get_id(), 'status': 'pending'}})

    return jsonify({"task_id": str(task_id)})

@app.route('/tasks/<jobId>', methods=['GET'])
def get_task(jobId):
    try:
        task = audio_tasks.find_one({'_id': ObjectId(jobId)})
        if task:
            return Response(json.dumps(task, default=json_util.default), mimetype='application/json')
        else:
            return jsonify({"error": "Task not found"}), 404
    except bson.errors.InvalidId:
        return jsonify({"error": "Invalid ObjectId format"}), 400

@app.route('/fetch-result/<taskId>', methods=['GET'])
def fetch_result(taskId):
    try:
        task = audio_tasks.find_one({'_id': ObjectId(taskId)})
        if task:
            if task.get('status') == 'completed':
                return jsonify({"status": "completed", "audio": task.get('audio')})
            else:
                return jsonify({"status": task.get('status')})
        else:
            return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/progress/<taskId>', methods=['GET'])
def get_progress(taskId):
    try:
        progress = redis_conn.get(f"progress_{taskId}")
        if progress:
            return jsonify({"progress": float(progress)})
        else:
            return jsonify({"progress": 0.0})  # Default to 0 if no progress found
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "mongodb": "down",
        "pytorch": "down",
        "redis": "down",
        "status": "down"
    }

    # Check MongoDB connection
    try:
        client.admin.command('ping')
        health_status["mongodb"] = "live"
    except Exception as e:
        print(f"MongoDB health check failed: {e}")

    # Check PyTorch
    if torch.cuda.is_available():
        print("PyTorch CUDA available")
        health_status["pytorch"] = "live"
    else:
        print("PyTorch CUDA not available")

    # Check Redis connection
    try:
        redis_conn.ping()
        print("Redis connection successful")
        health_status["redis"] = "live"
    except Exception as e:
        print(f"Redis health check failed: {e}")

    # Set the overall status
    if health_status["mongodb"] == "live" and health_status["pytorch"] == "live" and health_status["redis"] == "live":
        health_status["status"] = "live"


    print(f"Final health status: {health_status}")  # Debugging: print the health status

    return jsonify(health_status), 200 if health_status["status"] == "live" else 503


if __name__ == '__main__':
    app.run(debug=True, threaded=True)