"""
g4l_localhost.py - Simplified localhost backend for gary4juce
Removes WebSocket, Go service, MongoDB complexity while maintaining JUCE plugin compatibility
"""

import os
import json
import time
import uuid
import gc
import base64
import threading
import requests
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import torch
import redis
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, Field

# Import our audio processing functions (Gary/MusicGen only)
from g4laudio import process_audio, continue_music

# In both g4l_localhost.py AND localhost_melodyflow.py
import tempfile

# Use a consistent shared temp directory
SHARED_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gary4juce_shared")
os.makedirs(SHARED_TEMP_DIR, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__)
CORS(app)

# Redis connection (same as remote backend for compatibility)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# =============================================================================
# PYDANTIC MODELS (Keep existing models for validation)
# =============================================================================

class AudioRequest(BaseModel):
    audio_data: str
    model_name: str
    prompt_duration: int
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str
    model_name: Optional[str] = None
    prompt_duration: Optional[int] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None

class ContinueMusicRequest(BaseModel):
    session_id: Optional[str] = None
    model_name: Optional[str] = None
    prompt_duration: Optional[int] = None
    audio_data: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None

class TransformRequest(BaseModel):
    audio_data: Optional[str] = None
    variation: str
    session_id: Optional[str] = None
    flowstep: Optional[float] = Field(None, ge=0)
    solver: Optional[str] = None
    custom_prompt: Optional[str] = None

# =============================================================================
# GPU CLEANUP UTILITIES (Keep for memory management)
# =============================================================================

@contextmanager
def force_gpu_cleanup():
    """Enhanced GPU cleanup context manager."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

def clean_gpu_memory():
    """Utility function to force GPU memory cleanup."""
    if torch.cuda.is_available():
        devices = range(torch.cuda.device_count())
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

# =============================================================================
# SESSION MANAGEMENT (Redis-based, same format as remote backend)
# =============================================================================

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def store_session_data(session_id: str, data: dict):
    """Store session data in Redis with 1 hour expiration."""
    redis_client.setex(f"session:{session_id}", 3600, json.dumps(data))

def get_session_data(session_id: str):
    """Retrieve session data from Redis."""
    data = redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None

def store_session_progress(session_id: str, progress: int):
    """Store generation progress for polling."""
    redis_client.setex(f"progress:{session_id}", 3600, str(progress))

def get_session_progress(session_id: str):
    """Get current generation progress."""
    progress = redis_client.get(f"progress:{session_id}")
    return int(progress) if progress else 0

def store_session_status(session_id: str, status: str, error: str = None):
    """Store session status (processing, completed, failed)."""
    status_data = {"status": status}
    if error:
        status_data["error"] = error
    redis_client.setex(f"status:{session_id}", 3600, json.dumps(status_data))

def get_session_status(session_id: str):
    """Get session status."""
    status_data = redis_client.get(f"status:{session_id}")
    return json.loads(status_data) if status_data else {"status": "unknown"}

def store_audio_result(session_id: str, audio_base64: str):
    """Store generated audio result."""
    redis_client.setex(f"result:{session_id}", 3600, audio_base64)

def get_audio_result(session_id: str):
    """Get generated audio result."""
    return redis_client.get(f"result:{session_id}")

def store_original_audio(session_id: str, audio_base64: str):
    """Store original audio for undo functionality."""
    redis_client.setex(f"original:{session_id}", 3600, audio_base64)

def get_original_audio(session_id: str):
    """Get original audio for undo."""
    return redis_client.get(f"original:{session_id}")

def write_audio_to_temp_file(audio_base64, session_id):
    filename = f"input_{session_id}_{uuid.uuid4().hex[:8]}.wav"
    file_path = os.path.join(SHARED_TEMP_DIR, filename)  # Use shared directory
    
    audio_data = base64.b64decode(audio_base64)
    with open(file_path, 'wb') as f:
        f.write(audio_data)
    return file_path

def cleanup_temp_file(file_path):
    """Safely remove a temporary file."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up temp file {file_path}: {e}")

def store_queue_status_update(session_id: str, payload: dict):
    """Store queue/warmup/processing hints for the poller."""
    redis_client.setex(f"queue_status:{session_id}", 3600, json.dumps(payload))

def get_stored_queue_status(session_id: str):
    """Retrieve last queue/warmup/processing hint."""
    data = redis_client.get(f"queue_status:{session_id}")
    return json.loads(data) if data else None

# =============================================================================
# ADDITIONAL REDIS FUNCTIONS FOR LAST INPUT AUDIO
# =============================================================================

def store_last_input_audio(session_id: str, audio_base64: str):
    """Store last input audio for retry functionality."""
    redis_client.setex(f"last_input:{session_id}", 3600, audio_base64)

def get_last_input_audio(session_id: str):
    """Get last input audio for retry."""
    return redis_client.get(f"last_input:{session_id}")

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def run_audio_processing(session_id: str, audio_data: str, model_name: str,
                        prompt_duration: int, **kwargs):
    def progress_callback(current, total):
        progress_percent = int((current / total) * 100)
        store_session_progress(session_id, progress_percent)

        # FIRST nonzero progress => leave warming
        if progress_percent > 0:
            store_session_status(session_id, "processing")
            store_queue_status_update(session_id, {
                "status": "processing",
                "message": "generating…",
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

    def processing_thread():
        try:
            # Tell poller we are warming (this is BEFORE model load / HF download)
            store_session_status(session_id, "warming")
            store_session_progress(session_id, 0)
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": f'loading {model_name} (first run / hub download)',
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

            with force_gpu_cleanup():
                result_base64 = process_audio(
                    audio_data,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=kwargs.get('top_k', 250),
                    temperature=kwargs.get('temperature', 1.0),
                    cfg_coef=kwargs.get('cfg_coef', 3.0),
                    description=kwargs.get('description', '')
                )

                store_audio_result(session_id, result_base64)
                store_session_status(session_id, "completed")
                store_session_progress(session_id, 100)
                store_queue_status_update(session_id, {
                    "status": "completed",
                    "message": "done",
                    "source": "localhost"
                })
                print(f"[SUCCESS] Audio processing completed for {session_id}")

        except Exception as e:
            print(f"[ERROR] Audio processing failed for {session_id}: {e}")
            store_session_status(session_id, "failed", str(e))
            store_queue_status_update(session_id, {
                "status": "failed",
                "message": str(e),
                "source": "localhost"
            })
        finally:
            clean_gpu_memory()

    threading.Thread(target=processing_thread, daemon=True).start()

def run_continue_processing(session_id: str, audio_data: str, model_name: str,
                            prompt_duration: int, **kwargs):
    def progress_callback(current, total):
        progress_percent = int((current / total) * 100)
        store_session_progress(session_id, progress_percent)
        if progress_percent > 0:
            store_session_status(session_id, "processing")
            store_queue_status_update(session_id, {
                "status": "processing",
                "message": "generating…",
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

    def processing_thread():
        try:
            store_session_status(session_id, "warming")
            store_session_progress(session_id, 0)
            store_last_input_audio(session_id, audio_data)
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": f'loading {model_name} (first run / hub download)',
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

            with force_gpu_cleanup():
                result_base64 = continue_music(
                    audio_data,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=kwargs.get('top_k', 250),
                    temperature=kwargs.get('temperature', 1.0),
                    cfg_coef=kwargs.get('cfg_coef', 3.0),
                    description=kwargs.get('description', '')
                )

                store_audio_result(session_id, result_base64)
                store_session_status(session_id, "completed")
                store_session_progress(session_id, 100)
                store_queue_status_update(session_id, {
                    "status": "completed",
                    "message": "done",
                    "source": "localhost"
                })

        except Exception as e:
            print(f"[ERROR] Continue processing failed for {session_id}: {e}")
            store_session_status(session_id, "failed", str(e))
            store_queue_status_update(session_id, {
                "status": "failed",
                "message": str(e),
                "source": "localhost"
            })
        finally:
            clean_gpu_memory()

    threading.Thread(target=processing_thread, daemon=True).start()


# =============================================================================
# HTTP ENDPOINTS (Same interface as remote backend)
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint (matches remote backend format)."""
    health_status = {"status": "live", "service": "gary4juce-localhost"}

    # Check Redis (essential for session storage)
    try:
        redis_client.ping()
        health_status['redis'] = 'live'
    except Exception as e:
        health_status['redis'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check PyTorch/CUDA (essential for audio processing)
    try:
        if torch.cuda.is_available():
            health_status['pytorch'] = 'live'
            health_status['gpu'] = torch.cuda.get_device_name(0)
        else:
            health_status['pytorch'] = 'no GPU detected'
            health_status['status'] = 'degraded'
    except Exception as e:
        health_status['pytorch'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check audiocraft import (essential for Gary functionality)
    try:
        from g4laudio import process_audio
        health_status['audiocraft'] = 'live'
    except Exception as e:
        health_status['audiocraft'] = f'import error: {str(e)}'
        health_status['status'] = 'degraded'

    # Return appropriate status code
    status_code = 200 if health_status['status'] == 'live' else 500
    return jsonify(health_status), status_code

@app.route('/api/juce/process_audio', methods=['POST'])
def juce_process_audio():
    """Process audio - direct processing instead of queueing."""
    try:
        # Validate request
        request_data = AudioRequest(**request.json)
        session_id = generate_session_id()
        
        # Store session data (same format as remote)
        session_data = {
            'session_id': session_id,
            'model_name': request_data.model_name,
            'prompt_duration': request_data.prompt_duration,
            'parameters': {
                'top_k': request_data.top_k or 250,
                'temperature': request_data.temperature or 1.0,
                'cfg_coef': request_data.cfg_coef or 3.0,
            },
            'description': request_data.description or '',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        store_session_data(session_id, session_data)
        
        # Start processing immediately (no queue)
        run_audio_processing(
            session_id,
            request_data.audio_data,
            request_data.model_name,
            request_data.prompt_duration,
            top_k=request_data.top_k,
            temperature=request_data.temperature,
            cfg_coef=request_data.cfg_coef,
            description=request_data.description
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Audio processing started'
        })
        
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/juce/continue_music', methods=['POST'])
def juce_continue_music():
    """Continue music generation."""
    try:
        request_data = ContinueMusicRequest(**request.json)
        session_id = generate_session_id()
        
        # Store session data
        session_data = {
            'session_id': session_id,
            'model_name': request_data.model_name,
            'prompt_duration': request_data.prompt_duration,
            'type': 'continue',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        store_session_data(session_id, session_data)
        
        # Start continue processing
        run_continue_processing(
            session_id,
            request_data.audio_data,
            request_data.model_name,
            request_data.prompt_duration,
            top_k=request_data.top_k,
            temperature=request_data.temperature,
            cfg_coef=request_data.cfg_coef,
            description=request_data.description
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Continue processing started'
        })
        
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/juce/retry_music', methods=['POST'])
def juce_retry_music():
    """Retry music generation with new parameters - FIXED VERSION."""
    try:
        request_data = SessionRequest(**request.json)
        old_session_id = request_data.session_id
        new_session_id = generate_session_id()
        
        # Get original session data
        old_session_data = get_session_data(old_session_id)
        if not old_session_data:
            return jsonify({'success': False, 'error': 'Original session not found'}), 404
        
        # FIXED: Get last INPUT audio (not result audio!)
        last_input_audio = get_last_input_audio(old_session_id)
        if not last_input_audio:
            return jsonify({'success': False, 'error': 'No last input audio found for retry'}), 404
        
        # Create new session with updated parameters
        new_session_data = old_session_data.copy()
        new_session_data['session_id'] = new_session_id
        new_session_data['type'] = 'retry'
        new_session_data['original_session'] = old_session_id
        
        # Update with new parameters if provided
        if request_data.model_name:
            new_session_data['model_name'] = request_data.model_name
        if request_data.prompt_duration:
            new_session_data['prompt_duration'] = request_data.prompt_duration
        
        store_session_data(new_session_id, new_session_data)
        
        # FIXED: Start retry processing using last INPUT audio (not result audio)
        run_continue_processing(
            new_session_id,
            last_input_audio,  # Use the input audio that was used in the previous continuation
            new_session_data['model_name'],
            new_session_data['prompt_duration'],
            top_k=request_data.top_k,
            temperature=request_data.temperature,
            cfg_coef=request_data.cfg_coef,
            description=request_data.description
        )
        
        return jsonify({
            'success': True,
            'session_id': new_session_id,
            'message': 'Retry processing started'
        })
        
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/juce/undo_transform', methods=['POST'])
def juce_undo_transform():
    """Undo last transform by returning original audio."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
        
        # Get original audio
        original_audio = get_original_audio(session_id)
        if not original_audio:
            return jsonify({'success': False, 'error': 'No original audio found for undo'}), 404
        
        return jsonify({
            'success': True,
            'audio_data': original_audio,
            'message': 'Transform undone successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/juce/poll_status/<session_id>', methods=['GET'])
def juce_poll_status(session_id):
    try:
        status_data = get_session_status(session_id)      # {"status": "...", "error": "...?"}
        progress = get_session_progress(session_id)
        qstatus = get_stored_queue_status(session_id)     # may be None

        # Synthesize a warming hint if we look idle-but-working and nothing is stored yet
        if (status_data.get('status') in ('warming', 'processing') and progress == 0 and not qstatus):
            sess = get_session_data(session_id) or {}
            model_name = (sess.get('model_name') or 'model')
            qstatus = {
                "status": "warming",
                "message": f'loading {model_name} (first run / hub download)',
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "synthetic-localhost"
            }

        response = {
            "success": True,
            "status": status_data.get("status", "unknown"),
            "progress": progress,
            "queue_status": qstatus or {}
        }

        # Keep JUCE flags consistent with remote
        if status_data.get("status") in ("warming", "processing"):
            response["generation_in_progress"] = True
            response["transform_in_progress"] = False
        elif status_data.get("status") == "completed":
            response["generation_in_progress"] = False
            response["transform_in_progress"] = False
            audio_result = get_audio_result(session_id)
            if audio_result:
                response["audio_data"] = audio_result
        elif status_data.get("status") == "failed":
            response["generation_in_progress"] = False
            response["transform_in_progress"] = False
            response["error"] = status_data.get("error", "Unknown error")
        else:
            response["generation_in_progress"] = False
            response["transform_in_progress"] = False

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("🎵 Starting gary4juce localhost backend...")
    print("🔧 Redis connection:", "OK" if redis_client.ping() else "FAILED")
    print("🎯 CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)