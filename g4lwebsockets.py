from flask import Flask, request, copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room
import gevent
from pymongo import MongoClient
from gridfs import GridFS
import base64
import redis
from g4laudio import process_audio, continue_music, generate_session_id
from bson.objectid import ObjectId
from pydantic import BaseModel, ValidationError

# MongoDB setup
client = MongoClient('mongodb://localhost:27018/')
db = client['audio_generation_db']
sessions = db.sessions
fs = GridFS(db)

# Redis setup
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

app = Flask(__name__)
socketio = SocketIO(app, message_queue='redis://localhost:6379', async_mode='gevent', cors_allowed_origins="*", logger=True, engineio_logger=True, pingTimeout=240000, pingInterval=120000, max_http_buffer_size=16*1024*1024)

@app.route('/')
def index():
    return "The WebSocket server is running."

# Pydantic models for validation
class AudioRequest(BaseModel):
    audio_data: str
    model_name: str
    prompt_duration: int

class SessionRequest(BaseModel):
    session_id: str
    model_name: str = None
    prompt_duration: int = None

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
        request_data = AudioRequest(**data)
        session_id = generate_session_id()

        if is_generation_in_progress(session_id):
            emit('error', {'message': 'Generation already in progress', 'session_id': session_id}, room=session_id)
            return

        join_room(session_id)
        input_data_base64 = request_data.audio_data
        model_name = request_data.model_name
        prompt_duration = request_data.prompt_duration
        store_audio_data(session_id, input_data_base64, 'initial_audio')
        set_generation_in_progress(session_id, True)

        @copy_current_request_context
        def audio_processing_thread():
            try:
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100
                    emit('progress_update', {'progress': int(progress_percent), 'session_id': session_id}, room=session_id)

                result_base64 = process_audio(input_data_base64, model_name, progress_callback, prompt_duration=prompt_duration)
                store_audio_data(session_id, result_base64, 'last_processed_audio')
                emit('audio_processed', {'audio_data': result_base64, 'session_id': session_id}, room=session_id)
            except Exception as e:
                emit('error', {'message': str(e), 'session_id': session_id}, room=session_id)
            finally:
                set_generation_in_progress(session_id, False)

        gevent.spawn(audio_processing_thread)
    except ValidationError as e:
        emit('error', {'message': str(e), 'session_id': generate_session_id()})

@socketio.on('continue_music_request')
def handle_continue_music(data):
    try:
        request_data = SessionRequest(**data)
        session_id = request_data.session_id

        if is_generation_in_progress(session_id):
            emit('error', {'message': 'Generation already in progress', 'session_id': session_id}, room=session_id)
            return

        last_processed_base64 = retrieve_audio_data(session_id, 'last_processed_audio')
        if last_processed_base64 is None:
            emit('error', {'message': 'No last processed audio available for continuation', 'session_id': session_id}, room=session_id)
            return
        model_name = request_data.model_name or sessions.find_one({'_id': session_id}).get('model_name')
        prompt_duration = request_data.prompt_duration or sessions.find_one({'_id': session_id}).get('prompt_duration')
        set_generation_in_progress(session_id, True)

        @copy_current_request_context
        def continue_music_thread():
            try:
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100
                    emit('progress_update', {'progress': int(progress_percent), 'session_id': session_id}, room=session_id)
                
                result_base64 = continue_music(last_processed_base64, model_name, progress_callback, prompt_duration=prompt_duration)
                store_audio_data(session_id, last_processed_base64, 'last_input_audio')  # Store last input used for continuation
                store_audio_data(session_id, result_base64, 'last_processed_audio')  # Update last processed with new continuation
                emit('music_continued', {'audio_data': result_base64, 'session_id': session_id}, room=session_id)
            except Exception as e:
                emit('error', {'message': str(e), 'session_id': session_id}, room=session_id)
            finally:
                set_generation_in_progress(session_id, False)

        gevent.spawn(continue_music_thread)
    except ValidationError as e:
        emit('error', {'message': str(e), 'session_id': data.get('session_id')})

@socketio.on('retry_music_request')
def handle_retry_music(data):
    try:
        request_data = SessionRequest(**data)
        session_id = request_data.session_id

        if is_generation_in_progress(session_id):
            emit('error', {'message': 'Generation already in progress', 'session_id': session_id}, room=session_id)
            return
        last_input_base64 = retrieve_audio_data(session_id, 'last_input_audio')
        if last_input_base64 is None:
            emit('error', {'message': 'No last input audio available for retry', 'session_id': session_id}, room=session_id)
            return
        model_name = request_data.model_name or sessions.find_one({'_id': session_id}).get('model_name')
        prompt_duration = request_data.prompt_duration or sessions.find_one({'_id': session_id}).get('prompt_duration')
        set_generation_in_progress(session_id, True)

        @copy_current_request_context
        def retry_music_thread():
            try:
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100
                    emit('progress_update', {'progress': int(progress_percent), 'session_id': session_id}, room=session_id)

                result_base64 = continue_music(last_input_base64, model_name, progress_callback, prompt_duration=prompt_duration)
                store_audio_data(session_id, last_input_base64, 'last_input_audio')
                store_audio_data(session_id, result_base64, 'last_processed_audio')
                emit('music_retried', {'audio_data': result_base64, 'session_id': session_id}, room=session_id)
            except Exception as e:
                emit('error', {'message': str(e), 'session_id': session_id}, room=session_id)
            finally:
                set_generation_in_progress(session_id, False)

        gevent.spawn(retry_music_thread)
    except ValidationError as e:
        emit('error', {'message': str(e), 'session_id': data.get('session_id')})

@socketio.on('update_cropped_audio')
def handle_update_cropped_audio(data):
    try:
        session_id = data.get('session_id')  # Use get method to safely retrieve session_id
        audio_data_base64 = data.get('audio_data')  # Use get method to safely retrieve audio_data
        if session_id and audio_data_base64:
            store_audio_data(session_id, audio_data_base64, 'last_processed_audio')
            emit('update_cropped_audio_complete', {'message': 'Cropped audio updated', 'session_id': session_id}, room=session_id)
        else:
            raise ValueError("Missing session_id or audio_data")
    except Exception as e:
        emit('error', {'message': str(e), 'session_id': data.get('session_id')})

if __name__ == '__main__':
    socketio.run(app, debug=True)
