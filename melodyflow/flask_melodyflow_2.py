from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
import torchaudio
import time
import base64
import io
from audiocraft.models import MelodyFlow
import gc
import threading
from variations import VARIATIONS
import psutil
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import sys

# Add these imports at the top
import weakref
# Add these imports and keep existing ones
from contextlib import contextmanager

# Add these global variables
model_ref = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@contextmanager
def resource_cleanup():
    """Context manager to ensure proper cleanup of GPU resources."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

# Configure logging to filter out the specific error
class WebSocketErrorFilter(logging.Filter):
    def filter(self, record):
        return 'Cannot obtain socket from WSGI environment' not in str(record.msg)

# Add the filter to the root logger
logging.getLogger().addFilter(WebSocketErrorFilter())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('melodyflow')
handler = RotatingFileHandler('melodyflow.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(handler)

app = Flask(__name__)
app.start_time = time.time()  # Add this line right after app initialization
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_lock = threading.Lock()  # Lock for model access

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

def get_system_resources():
    """Monitor system resources."""
    try:
        # CPU and RAM
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # GPU memory using torch
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # Convert to MB

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'gpu_memory_used_mb': gpu_memory_used
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return None

def check_resource_availability():
    """Check if enough resources are available for processing."""
    resources = get_system_resources()
    if resources:
        # Define thresholds
        if (resources['cpu_percent'] > 90 or
            resources['memory_percent'] > 90 or
            resources['gpu_memory_used_mb'] > 10000):  # 10GB threshold
            raise AudioProcessingError(
                "Server is currently at capacity. Please try again later.",
                status_code=503
            )

@app.before_request
def before_request():
    """Check resources before processing requests."""
    if request.endpoint != 'health_check':  # Skip for health checks
        check_resource_availability()

@app.errorhandler(AudioProcessingError)
def handle_audio_processing_error(error):
    logger.error(f"Audio processing error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({'error': str(error)}), error.status_code

@app.errorhandler(Exception)
def handle_generic_error(error):
    logger.error(f"Unexpected error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# Audio Processing

def load_model():
    """Initialize the MelodyFlow model with improved memory management."""
    global model, model_ref
    with model_lock:
        if model is None:
            print("Loading MelodyFlow model...")
            model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs', device=DEVICE)
            # Create weak reference to track model
            model_ref = weakref.ref(model)
    return model

def load_audio_from_base64(audio_base64: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from base64 string."""
    try:
        # Decode base64 to binary
        audio_data = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_data)

        # Load audio
        waveform, sr = torchaudio.load(audio_file)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(device)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {str(e)}")

def save_audio_to_base64(waveform: torch.Tensor, sample_rate: int = 32000) -> str:
    """Convert audio tensor to base64 string."""
    try:
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform.cpu(), sample_rate, format="wav")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        raise AudioProcessingError(f"Failed to save audio: {str(e)}")

def process_audio(waveform: torch.Tensor, variation_name: str, session_id: str) -> torch.Tensor:
    """Process audio with selected variation using scoped model instantiation."""
    model = None
    try:
        if variation_name not in VARIATIONS:
            raise AudioProcessingError(f"Unknown variation: {variation_name}")
        config = VARIATIONS[variation_name]

        with resource_cleanup():
            # Initialize model within the processing function scope
            print("Loading MelodyFlow model...")
            model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs', device=device)

            # Find valid duration and get tokens
            max_valid_duration, tokens = find_max_duration(model, waveform)
            config['duration'] = max_valid_duration

            # Set model parameters
            model.set_generation_params(
                solver="euler",
                steps=config['steps'],
                duration=config['duration'],
            )

            model.set_editing_params(
                solver="euler",
                steps=config['steps'],
                target_flowstep=config['flowstep'],
                regularize=True,
                regularize_iters=2,
                keep_last_k_iters=1,
                lambda_kl=0.2,
            )

            def progress_callback(elapsed_steps: int, total_steps: int):
                progress = min((elapsed_steps / total_steps) * 100, 99.9)
                socketio.emit('progress', {
                    'progress': round(progress, 2),
                    'session_id': session_id
                })

            model._progress_callback = progress_callback

            edited_audio = model.edit(
                prompt_tokens=tokens,
                descriptions=[config['prompt']],
                src_descriptions=[""],
                progress=True,
                return_tokens=True
            )

            # Send 100% progress after processing is complete
            socketio.emit('progress', {
                'progress': 100.0,
                'session_id': session_id
            })

            return edited_audio[0][0]

    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")
    finally:
        # Explicitly delete the model
        if model is not None:
            del model
        with resource_cleanup():
            pass

def find_max_duration(model: MelodyFlow, waveform: torch.Tensor, sr: int = 32000, max_token_length: int = 750) -> tuple:
    """Binary search to find maximum duration that produces tokens under the limit."""
    min_seconds = 1
    max_seconds = waveform.shape[-1] / sr
    best_duration = min_seconds
    best_tokens = None

    while max_seconds - min_seconds > 0.1:
        mid_seconds = (min_seconds + max_seconds) / 2
        samples = int(mid_seconds * sr)
        test_waveform = waveform[..., :samples]

        try:
            tokens = model.encode_audio(test_waveform)
            token_length = tokens.shape[-1]

            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds

        except Exception as e:
            max_seconds = mid_seconds

    return best_duration, best_tokens

@app.route('/transform', methods=['POST'])
def transform_audio():
    """Handle audio transformation requests with improved memory management."""
    try:
        data = request.get_json()
        if not data or 'audio' not in data or 'variation' not in data or 'session_id' not in data:
            return jsonify({'error': 'Missing required data'}), 400

        with resource_cleanup():
            # Load and process audio
            input_waveform = load_audio_from_base64(data['audio'])
            processed_waveform = process_audio(input_waveform, data['variation'], data['session_id'])
            output_base64 = save_audio_to_base64(processed_waveform)
            
            # Explicitly delete intermediate tensors
            del input_waveform
            del processed_waveform

            return jsonify({
                'audio': output_base64,
                'message': 'Audio processed successfully'
            })

    except AudioProcessingError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/variations', methods=['GET'])
def get_variations():
    """Return list of available variations with CORS support."""
    try:
        variations_list = list(VARIATIONS.keys())
        variations_with_details = {
            name: {
                'prompt': VARIATIONS[name]['prompt'],
                'flowstep': VARIATIONS[name]['flowstep']
            } for name in variations_list
        }
        return jsonify({
            'variations': variations_with_details
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch variations: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with detailed system status."""
    try:
        resources = get_system_resources()
        model_loaded = model is not None
        gpu_available = torch.cuda.is_available()

        status = {
            'status': 'healthy' if model_loaded and gpu_available else 'degraded',
            'gpu_available': gpu_available,
            'model_loaded': model_loaded,
            'system_resources': resources,
            'uptime': time.time() - app.start_time,
            'version': os.getenv('MELODYFLOW_VERSION', 'dev'),
            'environment': os.getenv('FLASK_ENV', 'production')
        }

        return jsonify(status), 200 if status['status'] == 'healthy' else 503

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    # socketio.run(app, debug=False, host='0.0.0.0', port=8002)

    # Production mode with gevent-websocket (uncomment for production)
    # from gevent import pywsgi
    # from geventwebsocket.handler import WebSocketHandler
    # server = pywsgi.WSGIServer(('0.0.0.0', 8002), app, handler_class=WebSocketHandler)
    # server.serve_forever()

    # Production mode with waitress (currently shows websocket errors but everything works fine bro)
    
    from waitress import serve
    
    
    print("Starting MelodyFlow service in production mode...")
    print("Note: You may see a WebSocket environment message - this can be safely ignored as all functionality works correctly.")
    
    # Redirect stderr to filter out the specific error
    class StderrFilter:
        def write(self, text):
            if 'Cannot obtain socket from WSGI environment' not in text:
                sys.__stderr__.write(text)
        def flush(self):
            sys.__stderr__.flush()
    
    sys.stderr = StderrFilter()
    
    serve(app, host='0.0.0.0', port=8002, threads=4)