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
from pydub import AudioSegment
import io
from typing import Optional, Tuple

from rq import Queue, Retry
from redis import Redis

from pymongo import MongoClient, errors

from bson import ObjectId, json_util
import bson
import re

from g4laudio import continue_music
import gc
import time
from urllib.parse import urlparse, parse_qs

class YoutubeAudioProcessor:
    def __init__(self, cache_dir: str = '/dataset/gary'):
        self.cache_dir = cache_dir
        self.expected_sr = 32000
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
        os.makedirs(cache_dir, exist_ok=True)

    def get_segment_info(self, youtube_url: str) -> dict:
        """Get video duration and size information without downloading."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return {
                'duration': info.get('duration', 0),
                'filesize': info.get('filesize', 0)
            }

    def download_audio_segment(self, youtube_url: str, timestamp: float, duration: float = 30) -> str:
        """Download only the required segment of audio."""
        segment_file = os.path.join(
            self.cache_dir, 
            f"{base64.urlsafe_b64encode(f'{youtube_url}_{timestamp}_{duration}'.encode()).decode()}.mp3"
        )

        if os.path.exists(segment_file):
            return segment_file

        temp_file = os.path.join(self.cache_dir, 'temp_download.webm')
        
        # First, download the audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_file,
            'quiet': True,
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Then use ffmpeg to extract the segment
        start_time = str(timestamp)
        duration_str = str(duration)
        
        ffmpeg_command = [
            'ffmpeg', '-y',
            '-ss', start_time,
            '-t', duration_str,
            '-i', temp_file,
            '-acodec', 'libmp3lame',
            '-ar', '44100',
            '-ac', '2',
            '-b:a', '192k',
            segment_file
        ]
        
        try:
            import subprocess
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return segment_file

    def load_and_preprocess_audio(
        self, 
        file_path: str, 
        prompt_length: float
    ) -> Tuple[torch.Tensor, int]:
        """Load and preprocess the audio segment."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load audio in chunks to prevent memory issues
        audio_segment = AudioSegment.from_mp3(file_path)
        samples = torch.tensor(audio_segment.get_array_of_samples(), dtype=torch.float32)
        
        if audio_segment.channels == 2:
            samples = samples.view(-1, 2).t()
        else:
            samples = samples.unsqueeze(0)
        
        # Convert to proper format and sample rate
        samples = samples / (1 << (audio_segment.sample_width * 8 - 1))
        samples = samples.to(device)

        if audio_segment.frame_rate != self.expected_sr:
            resampler = torchaudio.transforms.Resample(
                audio_segment.frame_rate, 
                self.expected_sr
            ).to(device)
            samples = resampler(samples)

        # Ensure we only take the required prompt length
        prompt_frames = int(prompt_length * self.expected_sr)
        if samples.shape[1] > prompt_frames:
            samples = samples[:, :prompt_frames]

        return samples, self.expected_sr

    def process_youtube_audio(
        self, 
        youtube_url: str, 
        timestamp: float, 
        prompt_length: float
    ) -> Tuple[torch.Tensor, int]:
        """Main processing pipeline."""
        # First check video duration and estimate file size
        info = self.get_segment_info(youtube_url)
        
        if info['duration'] < timestamp:
            raise ValueError("Timestamp is beyond video duration")

        # Download only the segment we need
        segment_duration = prompt_length + 5  # Add a small buffer
        segment_file = self.download_audio_segment(
            youtube_url, 
            timestamp, 
            segment_duration
        )

        try:
            # Process the audio segment
            prompt_waveform, sr = self.load_and_preprocess_audio(
                segment_file, 
                prompt_length
            )
            return prompt_waveform, sr
        finally:
            # Clean up the segment file if it's not needed for caching
            if not self.should_cache(info['filesize']):
                os.remove(segment_file)

    def should_cache(self, filesize: int) -> bool:
        """Determine if the file should be cached based on size."""
        return filesize < self.max_file_size

# MongoDB connection with retry logic
def get_mongo_client():
    try:
        client = MongoClient('mongodb://mongo:27017/', serverSelectionTimeoutMS=60000)
        client.admin.command('ping')
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
youtube_processor = YoutubeAudioProcessor()

class YouTubeURLHandler:
    """Handles YouTube URL validation, normalization, and ID extraction."""
    
    # Supported YouTube domains
    DOMAINS = {'youtube.com', 'youtu.be', 'm.youtube.com', 'music.youtube.com', 'www.youtube.com'}
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extracts video ID from various YouTube URL formats.
        Returns None if no valid ID is found.
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            # Clean up the domain
            hostname = parsed.netloc.lower()
            if not any(domain in hostname for domain in YouTubeURLHandler.DOMAINS):
                return None
                
            # Handle youtu.be format
            if 'youtu.be' in hostname:
                return parsed.path.strip('/')
                
            # Handle various youtube.com formats
            if parsed.path.lower() in ['/watch', '/v/', '/embed/', '/shorts/']:
                # Get video ID from query parameters
                query = parse_qs(parsed.query)
                return query.get('v', [None])[0]
            
            # Handle direct paths (/v/{id}, /embed/{id}, /shorts/{id})
            path_parts = parsed.path.split('/')
            if len(path_parts) >= 3:
                return path_parts[2]
                
            return None
            
        except Exception:
            return None

    @staticmethod
    def normalize_url(url: str) -> Optional[str]:
        """
        Normalizes YouTube URL to standard format.
        Returns None if URL is invalid.
        """
        video_id = YouTubeURLHandler.extract_video_id(url)
        if not video_id:
            return None
        return f"https://youtube.com/watch?v={video_id}"

    @staticmethod
    def validate_video_id(video_id: str) -> bool:
        """Validates YouTube video ID format."""
        return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

    @staticmethod
    def process_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process a YouTube URL and return validation status, video ID, and normalized URL.
        
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (is_valid, video_id, normalized_url)
        """
        if not url:
            return False, None, None
            
        # Try to extract video ID
        video_id = YouTubeURLHandler.extract_video_id(url)
        if not video_id or not YouTubeURLHandler.validate_video_id(video_id):
            return False, None, None
            
        # Generate normalized URL
        normalized_url = f"https://youtube.com/watch?v={video_id}"
        return True, video_id, normalized_url

def is_valid_youtube_url(url: str) -> bool:
    """
    Backwards-compatible function for existing code.
    Returns True if URL is valid YouTube URL.
    """
    is_valid, _, _ = YouTubeURLHandler.process_url(url)
    return is_valid

def get_bpm(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
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

def generate_audio_continuation(prompt_waveform, sr, bpm, model, min_duration, max_duration, progress_callback=None):
    try:
        duration = calculate_duration(bpm, min_duration, max_duration)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            model_continue = MusicGen.get_pretrained(model)
            model_continue.set_custom_progress_callback(progress_callback)
            model_continue.set_generation_params(
                use_sampling=True, 
                top_k=150, 
                top_p=0.0, 
                temperature=1.0, 
                duration=duration, 
                cfg_coef=5
            )
            
            description = "drums, percussion"
            output = model_continue.generate_continuation(
                prompt_waveform, 
                prompt_sample_rate=sr, 
                descriptions=[description] if description else None,
                progress=True
            )
        return output.cpu().squeeze(0)
    except Exception as e:
        print(f"Error in generate_audio_continuation: {e}")
        raise

def save_generated_audio(output, sr):
    output_filename = 'generated_continuation'
    audio_write(output_filename, output, sr, strategy="loudness", loudness_compressor=True)
    return output_filename + '.wav'

def process_youtube_url(youtube_url, timestamp, model, promptLength, min_duration, max_duration, task_id):
    try:
        def progress_callback(current_step, total_steps):
            progress_percent = (current_step / total_steps) * 100
            print(f"Progress: {progress_percent}% for task {task_id}")
            redis_conn.set(f"progress_{task_id}", progress_percent, ex=600)

        # Use the new YouTube processor
        prompt_waveform, sr = youtube_processor.process_youtube_audio(
            youtube_url, 
            timestamp, 
            promptLength
        )

        # Get BPM from the downloaded segment
        segment_file = youtube_processor.download_audio_segment(youtube_url, timestamp, 30)
        bpm = get_bpm(segment_file)

        # Generate continuation
        output = generate_audio_continuation(
            prompt_waveform, 
            sr, 
            bpm, 
            model, 
            min_duration, 
            max_duration, 
            progress_callback
        )
        
        output_filename = save_generated_audio(output, sr)

        with open(output_filename, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        audio_tasks.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'output_filename': output_filename, 'status': 'completed', 'audio': encoded_audio}}
        )

        return output_filename
    except Exception as e:
        print(f"Error processing YouTube URL: {e}")
        audio_tasks.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'status': 'failed', 'error': str(e)}}
        )
        return None

def process_continuation(task_id, input_data_base64, musicgen_model, prompt_duration):
    try:
        def progress_callback(current_step, total_steps):
            progress_percent = (current_step / total_steps) * 100
            print(f"Progress: {progress_percent}% for task {task_id}")
            redis_conn.set(f"progress_{task_id}", progress_percent, ex=600)

        print(f"Memory before DB find: {torch.cuda.memory_allocated()} bytes")
        task = audio_tasks.find_one({'_id': ObjectId(task_id)})
        print(f"Memory after DB find: {torch.cuda.memory_allocated()} bytes")
        if not task:
            print("Task not found")
            return None

        output_data_base64 = continue_music(
            input_data_base64, 
            musicgen_model, 
            progress_callback=progress_callback, 
            prompt_duration=prompt_duration
        )
        
        task['audio'] = output_data_base64
        task['status'] = 'completed'

        print(f"Memory before DB update: {torch.cuda.memory_allocated()} bytes")
        audio_tasks.update_one({'_id': ObjectId(task_id)}, {"$set": task})
        print(f"Memory after DB update: {torch.cuda.memory_allocated()} bytes")

        return output_data_base64
    except Exception as e:
        print(f"Error processing continuation: {e}")
        audio_tasks.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'status': 'failed', 'error': str(e)}}
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

    min_duration = int(duration[0])
    max_duration = int(duration[1])

    is_valid, video_id, normalized_url = YouTubeURLHandler.process_url(youtube_url)
    if not is_valid:
        return jsonify({"error": "Invalid YouTube URL"}), 400
        
    # Use normalized URL for further processing
    youtube_url = normalized_url  # This ensures consistent URL format

    if not isinstance(timestamp, (int, float)) or timestamp < 0:
        return jsonify({"error": "Invalid timestamp"}), 400

    # Check video duration before proceeding
    try:
        info = youtube_processor.get_segment_info(youtube_url)
        if info['duration'] < timestamp:
            return jsonify({"error": "Timestamp is beyond video duration"}), 400
    except Exception as e:
        return jsonify({"error": f"Error accessing video: {str(e)}"}), 400

    audio_task = {
        'rq_job_id': None,
        'youtube_url': youtube_url,
        'timestamp': timestamp,
        'status': 'pending'
    }
    task_id = audio_tasks.insert_one(audio_task).inserted_id

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

    audio_tasks.update_one(
        {'_id': ObjectId(task_id)}, 
        {'$set': {'rq_job_id': job.get_id()}}
    )

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