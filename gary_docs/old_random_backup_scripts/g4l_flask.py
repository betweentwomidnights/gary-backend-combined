from flask import Flask, request, jsonify
import threading
from g4laudio import process_audio

app = Flask(__name__)

@app.route('/')
def index():
    return "The audio processing server is running."

@app.route('/process_audio', methods=['POST'])
def handle_audio_processing():
    data = request.get_json()
    input_data_base64 = data['audio_data']
    model_name = data['model_name']

    def audio_processing_thread(input_data_base64, model_name):
        try:
            print("Processing audio...")
            result_base64 = process_audio(input_data_base64, model_name)
            print("Audio processing completed.")
            return jsonify({'audio_data': result_base64})
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': str(e)}), 500

    # Run the audio processing in a separate thread
    thread = threading.Thread(target=audio_processing_thread, args=(input_data_base64, model_name))
    thread.start()

    return jsonify({'message': 'Audio processing started.'})

if __name__ == '__main__':
    app.run(debug=True)