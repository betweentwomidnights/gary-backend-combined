"""
Model warmup utility for gary-backend-combined
Provides an endpoint to warm up large models after VM restart
"""

import base64
import io
import numpy as np
from flask import Blueprint, jsonify
import torchaudio
from g4laudio import process_audio, AudioConfig
import torch

warmup_bp = Blueprint('warmup', __name__)

def generate_silent_audio(duration_seconds=6, sample_rate=32000):
    """Generate a short silent audio file for warmup testing"""
    num_samples = int(duration_seconds * sample_rate)
    # Create silent audio (stereo)
    audio = np.zeros((2, num_samples), dtype=np.float32)
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio)
    
    # Save to bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
    buffer.seek(0)
    
    # Return as base64
    return base64.b64encode(buffer.read()).decode('utf-8')

@warmup_bp.route('/api/warmup', methods=['POST'])
def warmup_models():
    """
    Warm up large models to avoid cold starts.
    Call this manually after VM restart: curl -X POST http://localhost:8000/api/warmup
    
    This will:
    1. Load hoenn_lofi (musicgen-large finetune)
    2. Load bleeps-medium (musicgen-medium)
    3. Run minimal generations to warm the page cache
    """
    try:
        results = {}
        
        # Generate a short silent audio clip for testing
        silent_audio_base64 = generate_silent_audio(duration_seconds=6)
        
        # Models to warm up (in order of size - largest first)
        models_to_warm = [
            {
                'name': 'thepatch/hoenn_lofi',
                'description': 'musicgen-large finetune',
                'duration': 30  # Very short generation just to load the model
            },
            {
                'name': 'thepatch/bleeps-medium',
                'description': 'musicgen-medium',
                'duration': 30
            }
        ]
        
        print("[WARMUP] Starting model warmup process...")
        
        for model_info in models_to_warm:
            model_name = model_info['name']
            print(f"[WARMUP] Loading and warming {model_name}...")
            
            try:
                # Use minimal settings for quick warmup
                config = AudioConfig(
                    prompt_duration=6.0,
                    output_duration=float(model_info['duration']),  # Very short
                    top_k=250,
                    temperature=1.0,
                    cfg_coef=3.0
                )
                
                # Run a quick generation (this loads the model and warms the cache)
                result = process_audio(
                    silent_audio_base64,
                    model_name,
                    progress_callback=None,  # No progress updates needed
                    prompt_duration=int(config.prompt_duration),
                    top_k=config.top_k,
                    temperature=config.temperature,
                    cfg_coef=config.cfg_coef,
                    description=None
                )
                
                results[model_name] = {
                    'status': 'success',
                    'message': f'Model warmed successfully',
                    'description': model_info['description']
                }
                print(f"[WARMUP] ✅ {model_name} warmed successfully")
                
            except Exception as model_error:
                results[model_name] = {
                    'status': 'failed',
                    'error': str(model_error),
                    'description': model_info['description']
                }
                print(f"[WARMUP] ❌ {model_name} failed: {model_error}")
        
        # Clean up GPU memory after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[WARMUP] Warmup process complete")
        
        return jsonify({
            'success': True,
            'message': 'Model warmup complete',
            'results': results
        })
        
    except Exception as e:
        print(f"[WARMUP] Error during warmup: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@warmup_bp.route('/api/warmup/status', methods=['GET'])
def warmup_status():
    """Simple endpoint to check if warmup is available"""
    return jsonify({
        'available': True,
        'models': [
            'thepatch/hoenn_lofi',
            'thepatch/bleeps-medium'
        ]
    })