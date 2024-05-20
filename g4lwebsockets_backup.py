from flask import Flask, copy_current_request_context
from flask_socketio import SocketIO, emit
import threading
from g4laudio import process_audio, continue_music  # Ensure this is the correct path to your g4laudio.py

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, max_http_buffer_size=16*1024*1024)

@app.route('/')
def index():
    return "The WebSocket server is running."

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to the server'})

@socketio.on('process_audio_request')
def handle_audio_processing(data):
    print("Received process_audio_request:", data)
    input_data_base64 = data['audio_data']
    model_name = data['model_name']

    @copy_current_request_context
    def audio_processing_thread(input_data_base64, model_name):
        try:
            print("Processing audio...")
            result_base64 = process_audio(input_data_base64, model_name)
            print("Emitting 'audio_processed' event...")
            emit('audio_processed', {'audio_data': result_base64})
        except Exception as e:
            print(f"An error occurred: {e}")
            emit('error', {'message': str(e)})

    threading.Thread(target=audio_processing_thread, args=(input_data_base64, model_name)).start()

@socketio.on('continue_music_request')
def handle_continue_music(data):
    print("Received continue_music_request:", data)
    input_data_base64 = data['audio_data']
    model_name = data['model_name']

    @copy_current_request_context
    def continue_music_thread(input_data_base64, model_name):
        try:
            print("Continuing music...")
            result_base64 = continue_music(input_data_base64, model_name)
            print("Emitting 'music_continued' event...")
            emit('music_continued', {'audio_data': result_base64})
        except Exception as e:
            print(f"An error occurred: {e}")
            emit('error', {'message': str(e)})

    threading.Thread(target=continue_music_thread, args=(input_data_base64, model_name)).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
