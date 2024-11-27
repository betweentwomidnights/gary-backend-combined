import gevent.monkey
gevent.monkey.patch_all()

from flask import Flask, jsonify, request, copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room
import gevent
from pymongo import MongoClient
from gridfs import GridFS
import base64
import redis
from g4laudio import process_audio, continue_music, generate_session_id
from bson.objectid import ObjectId
from pydantic import BaseModel, ValidationError
import torch
from flask_cors import CORS  # Import CORS
import json

from typing import Optional
# MongoDB setup
# THIS IS THE LOCAL VERSION
# client = MongoClient('mongodb://localhost:27017/')
client = MongoClient('mongodb://mongo:27017/')
db = client['audio_generation_db']
sessions = db.sessions
fs = GridFS(db)

# Redis setup

# THIS IS THE LOCAL VERSION
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
# redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# THIS IS THE LOCAL VERSION
socketio = SocketIO(
    app,
    message_queue='redis://redis:6379',
    async_mode='gevent',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    ping_timeout=240,  # Use snake_case
    ping_interval=120,  # Use snake_case
    max_http_buffer_size=64*1024*1024
)

@app.route('/')
def index():
    return "The WebSocket server is running."

# Pydantic models for validation
class AudioRequest(BaseModel):
    audio_data: str
    model_name: str
    prompt_duration: int
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None  # New optional field

class SessionRequest(BaseModel):
    session_id: str
    model_name: Optional[str] = None
    prompt_duration: Optional[int] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None  # New optional field

class ContinueMusicRequest(BaseModel):
    session_id: str
    model_name: Optional[str] = None
    prompt_duration: Optional[int] = None
    audio_data: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None  # New optional field

def store_audio_in_gridfs(data, filename):
    """Store audio data in GridFS."""
    audio_data = base64.b64decode(data)
    file_id = fs.put(audio_data, filename=filename)
    return str(file_id)

def retrieve_audio_from_gridfs(file_id):
    """Retrieve audio data from GridFS."""
    try:
        file = fs.get(ObjectId(file_id))
        return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error retrieving audio from GridFS: {e}")
        return None

def store_audio_data(session_id, audio_data, key):
    """Store session data in MongoDB with GridFS."""
    file_id = store_audio_in_gridfs(audio_data, f"{session_id}_{key}.wav")
    sessions.update_one({'_id': session_id}, {'$set': {key: file_id}}, upsert=True)

def retrieve_audio_data(session_id, key):
    """Retrieve specific audio data from MongoDB."""
    session_data = sessions.find_one({'_id': session_id})
    file_id = session_data.get(key) if session_data else None
    return retrieve_audio_from_gridfs(file_id) if file_id else None

def set_generation_in_progress(session_id, in_progress):
    """Set or unset the generation_in_progress flag in Redis."""
    redis_client.set(f"{session_id}_generation_in_progress", str(in_progress))

def is_generation_in_progress(session_id):
    """Check if generation is in progress using Redis."""
    return redis_client.get(f"{session_id}_generation_in_progress") == b'True'

@socketio.on('cleanup_session_request')
def handle_cleanup_request(data):
    try:
        request_data = SessionRequest(**data)
        session_id = request_data.session_id
        if session_id:
            sessions.delete_one({'_id': session_id})
            leave_room(session_id)
            redis_client.delete(f"{session_id}_generation_in_progress")
            emit('cleanup_complete', {'message': 'Session cleaned up', 'session_id': session_id}, room=session_id)
    except ValidationError as e:
        emit('error', {'message': str(e), 'session_id': data.get('session_id')})

@socketio.on('process_audio_request')
def handle_audio_processing(data):
    try:
        # Check if the received data is a string (raw JSON string from Swift)
        if isinstance(data, str):
            # Remove both single and double backslashes
            clean_data = data.replace("\\\\", "\\").replace("\\", "")
            # Parse the cleaned raw JSON string into a dictionary
            try:
                data = json.loads(clean_data)
            except json.JSONDecodeError as e:
                emit('error', {'message': 'Invalid JSON format: ' + str(e)})
                return

        # Clean model_name and strip leading/trailing spaces
        if "model_name" in data:
            data["model_name"] = data["model_name"].replace("\\", "").strip()

        # Clean optional parameters and strip leading/trailing spaces
        for param in ['top_k', 'temperature', 'cfg_coef', 'description']:
            if param in data and data[param] is not None:
                data[param] = str(data[param]).replace("\\", "").strip()

        # Proceed with the usual flow
        request_data = AudioRequest(**data)
        session_id = generate_session_id()

        if is_generation_in_progress(session_id):
            emit('error', {'message': 'Generation already in progress', 'session_id': session_id}, room=session_id)
            return

        join_room(session_id)
        input_data_base64 = request_data.audio_data
        model_name = request_data.model_name
        prompt_duration = request_data.prompt_duration

        # Extract optional parameters with default values if not provided
        top_k = int(request_data.top_k) if request_data.top_k is not None else 250
        temperature = float(request_data.temperature) if request_data.temperature is not None else 1.0
        cfg_coef = float(request_data.cfg_coef) if request_data.cfg_coef is not None else 3.0
        description = request_data.description if request_data.description else None

        # Log relevant information without base64 data
        print(f"Received process_audio_request for session {session_id} with model_name: {model_name}, prompt_duration: {prompt_duration}")

        store_audio_data(session_id, input_data_base64, 'initial_audio')
        set_generation_in_progress(session_id, True)

        @copy_current_request_context
        def audio_processing_thread():
            try:
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100
                    emit('progress_update', {'progress': int(progress_percent), 'session_id': session_id}, room=session_id)

                # Call process_audio with new parameters
                result_base64 = process_audio(
                    input_data_base64,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=top_k,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    description=description
                )
                print(f"Audio processed successfully for session {session_id}")

                store_audio_data(session_id, result_base64, 'last_processed_audio')
                emit('audio_processed', {'audio_data': result_base64, 'session_id': session_id}, room=session_id)
            except Exception as e:
                print(f"Error during audio processing thread for session {session_id}: {e}")
                emit('error', {'message': str(e), 'session_id': session_id})
            finally:
                set_generation_in_progress(session_id, False)

        gevent.spawn(audio_processing_thread)

    except ValidationError as e:
        emit('error', {'message': str(e), 'session_id': generate_session_id()})

@socketio.on('continue_music_request')
def handle_continue_music(data):
    try:
        # Check if the received data is a string (raw JSON string from Swift)
        if isinstance(data, str):
            # Remove both single and double backslashes
            clean_data = data.replace("\\\\", "\\").replace("\\", "")
            # Parse the cleaned raw JSON string into a dictionary
            try:
                data = json.loads(clean_data)
            except json.JSONDecodeError as e:
                emit('error', {'message': 'Invalid JSON format: ' + str(e)})
                return

        # Clean model_name and strip leading/trailing spaces
        if "model_name" in data:
            data["model_name"] = data["model_name"].replace("\\", "").strip()

        # Clean session_id
        if "session_id" in data:
            data["session_id"] = data["session_id"].replace("\\", "").strip()

        # Clean optional parameters and strip leading/trailing spaces
        for param in ['top_k', 'temperature', 'cfg_coef', 'description']:
            if param in data and data[param] is not None:
                data[param] = str(data[param]).replace("\\", "").strip()

        # Proceed with the usual flow using the updated Pydantic model
        request_data = ContinueMusicRequest(**data)
        session_id = request_data.session_id

        if is_generation_in_progress(session_id):
            emit('error', {'message': 'Generation already in progress', 'session_id': session_id}, room=session_id)
            return

        # Use 'audio_data' from data if available, else retrieve from session
        if request_data.audio_data:
            input_data_base64 = request_data.audio_data
            print(f"Using 'audio_data' from request for session {session_id}")
        else:
            input_data_base64 = retrieve_audio_data(session_id, 'last_processed_audio')
            print(f"Retrieved 'last_processed_audio' from session {session_id}")

        if input_data_base64 is None:
            emit('error', {'message': 'No audio data available for continuation', 'session_id': session_id}, room=session_id)
            return

        # Extract optional parameters with default values if not provided
        top_k = int(request_data.top_k) if request_data.top_k is not None else 250
        temperature = float(request_data.temperature) if request_data.temperature is not None else 1.0
        cfg_coef = float(request_data.cfg_coef) if request_data.cfg_coef is not None else 3.0
        description = request_data.description if request_data.description else None

        model_name = request_data.model_name or sessions.find_one({'_id': session_id}).get('model_name')
        prompt_duration = request_data.prompt_duration or sessions.find_one({'_id': session_id}).get('prompt_duration')

        print(f"Continuing music for session {session_id} with model_name: {model_name}, prompt_duration: {prompt_duration}")

        set_generation_in_progress(session_id, True)

        @copy_current_request_context
        def continue_music_thread():
            try:
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100
                    emit('progress_update', {'progress': int(progress_percent), 'session_id': session_id}, room=session_id)

                result_base64 = continue_music(
                    input_data_base64,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=top_k,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    description=description
                )

                store_audio_data(session_id, input_data_base64, 'last_input_audio')
                store_audio_data(session_id, result_base64, 'last_processed_audio')

                # Calculate the size of the base64 string in bytes
                result_size_bytes = len(result_base64.encode('utf-8'))

                # Get the max_http_buffer_size (ensure it's consistent with your SocketIO configuration)
                max_size_bytes = 64 * 1024 * 1024  # 64 MB

                if result_size_bytes > max_size_bytes:
                    emit('error', {
                        'message': 'Generated audio data is too large to send.',
                        'session_id': session_id,
                        'code': 'DATA_TOO_LARGE'
                    }, room=session_id)
                    print(f"Generated audio data is too large for session {session_id}: {result_size_bytes} bytes.")
                else:
                    try:
                        emit('music_continued', {'audio_data': result_base64, 'session_id': session_id}, room=session_id)
                    except Exception as e:
                        print(f"Error emitting music_continued for session {session_id}: {e}")
                        emit('error', {
                            'message': 'Error sending generated audio data.',
                            'session_id': session_id,
                            'code': 'EMIT_ERROR'
                        }, room=session_id)
            except Exception as e:
                print(f"Error during continue_music_thread for session {session_id}: {e}")
                emit('error', {'message': str(e), 'session_id': session_id}, room=session_id)
            finally:
                set_generation_in_progress(session_id, False)

        gevent.spawn(continue_music_thread)
    except ValidationError as e:
        emit('error', {'message': str(e), 'session_id': data.get('session_id') if isinstance(data, dict) else None})
        
@socketio.on('retry_music_request')
def handle_retry_music(data):
    try:
        # Check if the received data is a string (raw JSON string from Swift)
        if isinstance(data, str):
            # Remove both single and double backslashes
            clean_data = data.replace("\\\\", "\\").replace("\\", "")
            # Parse the cleaned raw JSON string into a dictionary
            try:
                data = json.loads(clean_data)
            except json.JSONDecodeError as e:
                emit('error', {'message': 'Invalid JSON format: ' + str(e)})
                return

        # Clean model_name and strip leading/trailing spaces
        if "model_name" in data:
            data["model_name"] = data["model_name"].replace("\\", "").strip()

        # Clean session_id
        if "session_id" in data:
            data["session_id"] = data["session_id"].replace("\\", "").strip()

        # Clean optional parameters and strip leading/trailing spaces
        for param in ['top_k', 'temperature', 'cfg_coef', 'description']:
            if param in data and data[param] is not None:
                data[param] = str(data[param]).replace("\\", "").strip()

        # Proceed with the usual flow using the updated data
        request_data = SessionRequest(**data)
        session_id = request_data.session_id

        if is_generation_in_progress(session_id):
            emit('error', {'message': 'Generation already in progress', 'session_id': session_id}, room=session_id)
            return

        last_input_base64 = retrieve_audio_data(session_id, 'last_input_audio')
        if last_input_base64 is None:
            emit('error', {'message': 'No last input audio available for retry', 'session_id': session_id}, room=session_id)
            return

        # Extract optional parameters with default values if not provided
        top_k = int(request_data.top_k) if request_data.top_k is not None else 250
        temperature = float(request_data.temperature) if request_data.temperature is not None else 1.0
        cfg_coef = float(request_data.cfg_coef) if request_data.cfg_coef is not None else 3.0
        description = request_data.description if request_data.description else None

        model_name = request_data.model_name or sessions.find_one({'_id': session_id}).get('model_name')
        prompt_duration = request_data.prompt_duration or sessions.find_one({'_id': session_id}).get('prompt_duration')

        print(f"Retrying music for session {session_id} with model_name: {model_name}, prompt_duration: {prompt_duration}")

        set_generation_in_progress(session_id, True)

        @copy_current_request_context
        def retry_music_thread():
            try:
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100
                    emit('progress_update', {'progress': int(progress_percent), 'session_id': session_id}, room=session_id)

                result_base64 = continue_music(
                    last_input_base64,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=top_k,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    description=description
                )
                store_audio_data(session_id, last_input_base64, 'last_input_audio')
                store_audio_data(session_id, result_base64, 'last_processed_audio')
                emit('music_retried', {'audio_data': result_base64, 'session_id': session_id}, room=session_id)
            except Exception as e:
                print(f"Error during retry_music_thread for session {session_id}: {e}")
                emit('error', {'message': str(e), 'session_id': session_id}, room=session_id)
            finally:
                set_generation_in_progress(session_id, False)

        gevent.spawn(retry_music_thread)
    except ValidationError as e:
        session_id = data.get('session_id') if isinstance(data, dict) else None
        emit('error', {'message': str(e), 'session_id': session_id})

@socketio.on('update_cropped_audio')
def handle_update_cropped_audio(data):
    try:
        # Check if the received data is a string (raw JSON string from Swift)
        if isinstance(data, str):
            # Remove both single and double backslashes
            clean_data = data.replace("\\\\", "\\").replace("\\", "")
            # Parse the cleaned raw JSON string into a dictionary
            try:
                data = json.loads(clean_data)
            except json.JSONDecodeError as e:
                emit('error', {'message': 'Invalid JSON format: ' + str(e)})
                return

        # Proceed with the usual flow
        request_data = SessionRequest(**data)
        session_id = request_data.session_id
        audio_data_base64 = data.get('audio_data')  # Use get method to safely retrieve audio_data

        if session_id and audio_data_base64:
            store_audio_data(session_id, audio_data_base64, 'last_processed_audio')
            emit('update_cropped_audio_complete', {'message': 'Cropped audio updated', 'session_id': session_id}, room=session_id)
            print(f"Cropped audio updated for session {session_id}")
        else:
            raise ValueError("Missing session_id or audio_data")
    except Exception as e:
        session_id = data.get('session_id') if isinstance(data, dict) else 'unknown'
        print(f"Error in update_cropped_audio for session {session_id}: {e}")
        emit('error', {'message': str(e), 'session_id': session_id})

# Robust Health Check Route
@app.route('/health', methods=['GET'])
def health_check():
    health_status = {"status": "live"}

    # Check MongoDB
    try:
        client.admin.command('ping')
        health_status['mongodb'] = 'live'
    except Exception as e:
        health_status['mongodb'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check Redis
    try:
        redis_client.ping()
        health_status['redis'] = 'live'
    except Exception as e:
        health_status['redis'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check PyTorch (Optional, if it's critical)
    try:
        if torch.cuda.is_available():
            health_status['pytorch'] = 'live'
        else:
            health_status['pytorch'] = 'no GPU detected'
    except Exception as e:
        health_status['pytorch'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    return jsonify(health_status), 200 if health_status['status'] == 'live' else 500

if __name__ == '__main__':
    socketio.run(app, debug=False)