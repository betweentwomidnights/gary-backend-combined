from flask import Flask, copy_current_request_context
from flask_socketio import SocketIO, emit
import threading
from pymongo import MongoClient
from bson.objectid import ObjectId
from g4laudio import process_audio, continue_music, generate_session_id  # Use the imported functions

# MongoDB setup
client = MongoClient('mongodb://localhost:27018/')  # Ensure the port is correctly specified
db = client['audio_generation_db']  # A separate database for audio generation tasks
sessions = db.sessions  # A separate collection for session data

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, max_http_buffer_size=16*1024*1024)

@app.route('/')
def index():
    return "The WebSocket server is running."

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to the server'})

def store_audio_data(session_id, audio_data, model_name):
    """Store session data in MongoDB."""
    sessions.insert_one({
        '_id': session_id,
        'audio_data': audio_data,
        'model_name': model_name
    })

def retrieve_audio_data(session_id):
    """Retrieve session data from MongoDB."""
    return sessions.find_one({'_id': session_id})

@socketio.on('cleanup_session_request')
def handle_cleanup_request(data):
    session_id = data['session_id']
    if session_id:
        sessions.delete_one({'_id': session_id})
        print(f"Session {session_id} and associated data have been cleaned up.")

@socketio.on('process_audio_request')
def handle_audio_processing(data):
    # Always generate a new session ID for a fresh start
    session_id = generate_session_id()
    input_data_base64 = data['audio_data']
    model_name = data['model_name']
    store_audio_data(session_id, input_data_base64, model_name)  # Store the new session data

    @copy_current_request_context
    def audio_processing_thread(input_data_base64, model_name, session_id):
        try:
            print("Processing audio with new session...");
            result_base64 = process_audio(input_data_base64, model_name)
            print("Emitting 'audio_processed' event with new session ID...");
            emit('audio_processed', {'audio_data': result_base64, 'session_id': session_id})
        except Exception as e:
            print(f"An error occurred: {e}")
            emit('error', {'message': str(e), 'session_id': session_id})

    threading.Thread(target=audio_processing_thread, args=(input_data_base64, model_name, session_id)).start()


@socketio.on('continue_music_request')
def handle_continue_music(data):
    session_id = data.get('session_id', generate_session_id())
    input_data_base64 = data['audio_data']
    model_name = data['model_name']

    # Store or update the input data in MongoDB immediately before processing
    store_or_update_audio_data(session_id, input_data_base64, model_name)

    @copy_current_request_context
    def continue_music_thread(input_data_base64, model_name, session_id):
        try:
            print("Continuing music...")
            result_base64 = continue_music(input_data_base64, model_name)
            print("Emitting 'music_continued' event...")
            emit('music_continued', {'audio_data': result_base64, 'session_id': session_id})
        except Exception as e:
            print(f"An error occurred: {e}")
            emit('error', {'message': str(e), 'session_id': session_id})

    threading.Thread(target=continue_music_thread, args=(input_data_base64, model_name, session_id)).start()

def store_or_update_audio_data(session_id, audio_data, model_name):
    """Store or update the session data in MongoDB."""
    sessions.update_one(
        {'_id': session_id},
        {'$set': {'audio_data': audio_data, 'model_name': model_name}},
        upsert=True
    )


@socketio.on('retry_music_request')
def handle_retry_music(data):
    session_id = data['session_id']
    session_data = retrieve_audio_data(session_id)
    if session_data:
        input_data_base64 = session_data['audio_data']
        model_name = data.get('model_name', session_data['model_name'])  # Use the model name from the request if provided

        @copy_current_request_context
        def retry_music_thread(input_data_base64, model_name, session_id):
            try:
                print("Retrying music generation with model:", model_name)
                result_base64 = continue_music(input_data_base64, model_name)
                print("Emitting 'music_retried' event...")
                emit('music_retried', {'audio_data': result_base64, 'session_id': session_id})
            except Exception as e:
                print(f"An error occurred: {e}")
                emit('error', {'message': str(e), 'session_id': session_id})

        threading.Thread(target=retry_music_thread, args=(input_data_base64, model_name, session_id)).start()
    else:
        emit('error', {'message': 'Session data not found for retry', 'session_id': session_id})

if __name__ == '__main__':
    socketio.run(app, debug=True)