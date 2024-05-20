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
import json
import librosa
import soundfile as sf
from rq import Queue
from redis import Redis
from pymongo import MongoClient
from bson import ObjectId, json_util
import bson
from flask import Response
from urllib.parse import urlparse
import re
import commune as c

class CommieGary(c.Module):
    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        CORS(self.app)
        self.executor = ThreadPoolExecutor(max_workers=24)
        self.client = MongoClient('mongodb://localhost:27018/')
        self.db = self.client['name']
        self.audio_tasks = self.db.audio_tasks
        self.redis_conn = Redis(host='localhost', port=6379)
        self.q = Queue(connection=self.redis_conn)
        self.module_path = 'commie_gary'

    def is_valid_youtube_url(self, url):
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?\s]{11})')
        youtube_pattern = re.compile(youtube_regex)
        return re.match(youtube_pattern, url) is not None

    def cleanup_files(self, *file_paths):
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

    def download_audio(self, youtube_url):
        downloaded_mp3 = 'downloaded_audio.mp3'
        downloaded_webm = 'downloaded_audio.webm'
        self.cleanup_files(downloaded_mp3, downloaded_webm)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': 'downloaded_audio.%(ext)s',
            'keepvideo': True,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return downloaded_mp3, downloaded_webm

    def get_bpm(self, downloaded_mp3):
        audio, sr = librosa.load(downloaded_mp3, sr=None)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        if 120 < tempo < 200:
            tempo = tempo / 2
        return tempo

    def calculate_duration(self, bpm, min_duration, max_duration):
        single_bar_duration = 4 * 60 / bpm
        bars = max(min_duration // single_bar_duration, 1)
        
        while single_bar_duration * bars < min_duration:
            bars += 1
        
        duration = single_bar_duration * bars
        
        while duration > max_duration and bars > 1:
            bars -= 1
            duration = single_bar_duration * bars
        
        return duration

    def load_and_preprocess_audio(self, file_path, timestamp, promptLength):
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

    def generate_audio_continuation(self, prompt_waveform, sr, bpm, model, min_duration, max_duration):
        # Calculate the duration to end at a bar
        duration = self.calculate_duration(bpm, min_duration, max_duration)

        model_continue = MusicGen.get_pretrained(model)
        model_continue.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=duration, cfg_coef=3)
        output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)
        return output.cpu().squeeze(0)

    def save_generated_audio(self, output, sr):
        output_filename = 'generated_continuation'
        audio_write(output_filename, output, sr, strategy="loudness", loudness_compressor=True)
        return output_filename + '.wav'

    def process_youtube_url(self, youtube_url, timestamp, model, promptLength, min_duration, max_duration):
        try:
            downloaded_mp3, downloaded_webm = self.download_audio(youtube_url)
            bpm = self.get_bpm(downloaded_mp3)
            prompt_waveform, sr = self.load_and_preprocess_audio(downloaded_mp3, timestamp, promptLength)
            output = self.generate_audio_continuation(prompt_waveform, sr, bpm, model, min_duration, max_duration)
            output_filename = self.save_generated_audio(output, sr)

            # Encode the audio data
            with open(output_filename, 'rb') as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

            # Save task info, audio reference, and status in MongoDB
            self.audio_tasks.update_one(
                {'youtube_url': youtube_url, 'timestamp': timestamp},
                {'$set': {'output_filename': output_filename, 'status': 'completed', 'audio': encoded_audio}}
            )

            self.cleanup_files(downloaded_mp3, downloaded_webm)
            return output_filename
        except Exception as e:
            print(f"Error processing YouTube URL: {e}")
            # Update the task status in MongoDB in case of an error
            self.audio_tasks.update_one(
                {'youtube_url': youtube_url, 'timestamp': timestamp},
                {'$set': {'status': 'failed'}}
            )
            return None

    @self.app.route('/generate', methods=['POST'])
    def generate_audio(self):
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
        if not self.is_valid_youtube_url(youtube_url):
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Validate timestamp
        if not isinstance(timestamp, (int, float)) or timestamp < 0:
            return jsonify({"error": "Invalid timestamp"}), 400

        # Enqueue the task
        job = self.q.enqueue(self.process_youtube_url, youtube_url, timestamp, model, promptLength, min_duration, max_duration, job_timeout=600)
        
        # Save task info in MongoDB
        audio_task = {
            'rq_job_id': job.get_id(),
            'youtube_url': youtube_url,
            'timestamp': timestamp,
            'status': 'pending'
        }
        task_id = self.audio_tasks.insert_one(audio_task).inserted_id

        return jsonify({"task_id": str(task_id)})

    @self.app.route('/tasks/<jobId>', methods=['GET'])
    def get_task(self, jobId):
        try:
            task = self.audio_tasks.find_one({'_id': ObjectId(jobId)})
            if task:
                return Response(json.dumps(task, default=json_util.default), mimetype='application/json')
            else:
                return jsonify({"error": "Task not found"}), 404
        except bson.errors.InvalidId:
            return jsonify({"error": "Invalid ObjectId format"}), 400

    def serve(self):
        self.app.run(debug=True, threaded=True)

commie_gary = CommieGary()
c.register(commie_gary)