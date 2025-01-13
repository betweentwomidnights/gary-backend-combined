import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
import base64
import io
import uuid
import torchaudio.transforms as T
import gc
from typing import Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors with detailed messages."""
    pass

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    prompt_duration: float = 6.0
    top_k: int = 250
    temperature: float = 1.0
    cfg_coef: float = 3.0
    target_sr: int = 32000
    output_duration: float = 30.0

@contextmanager
def resource_cleanup():
    """Context manager to ensure proper cleanup of GPU resources."""
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# Add this helper at the top of the file
def get_device():
    """Get the most appropriate device available."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

# Audio preprocessing utility functions
def peak_normalize(y, target_peak=0.9):
    """Normalize audio to a target peak amplitude."""
    return target_peak * (y / np.max(np.abs(y)))

def rms_normalize(y, target_rms=0.05):
    """Normalize audio to a target RMS value."""
    current_rms = np.sqrt(np.mean(y**2))
    return y * (target_rms / current_rms)

def preprocess_audio(waveform):
    """Preprocess audio waveform."""
    device = get_device()
    waveform_np = waveform.cpu().squeeze().numpy()
    processed_waveform_np = waveform_np
    return torch.from_numpy(processed_waveform_np).unsqueeze(0).to(device)

def wrap_audio_if_needed(waveform, sr, desired_duration):
    """Wrap audio if needed to match desired duration."""
    current_duration = waveform.shape[-1] / sr
    
    # If the current duration is already longer than or equal to the desired duration, return as is
    if current_duration >= desired_duration:
        return waveform

    # Calculate how much silence is needed (in samples)
    padding_duration = desired_duration - current_duration
    padding_samples = int(padding_duration * sr)
    
    # Create a tensor of zeros (silence) with the necessary number of samples
    silence = torch.zeros(1, padding_samples).to(waveform.device)  # Ensure it matches the device (GPU/CPU)
    
    # Append the silence to the original waveform
    padded_waveform = torch.cat([waveform, silence], dim=-1)
    
    return padded_waveform

def load_and_validate_audio(input_data_base64: str) -> tuple[torch.Tensor, int]:
    """Load and validate input audio from base64 string."""
    device = get_device()
    input_audio = None
    try:
        input_data = base64.b64decode(input_data_base64)
        input_audio = io.BytesIO(input_data)
        song, sr = torchaudio.load(input_audio)
        if song.size(0) == 0 or song.size(1) == 0:
            raise AudioProcessingError("Input audio is empty")
        return song.to(device), sr
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    finally:
        if input_audio is not None:
            input_audio.close()

def save_audio_to_base64(waveform: torch.Tensor, sample_rate: int) -> str:
    """Save audio tensor to base64 string with proper resource cleanup."""
    output_audio = None
    try:
        output_audio = io.BytesIO()
        torchaudio.save(output_audio, format='wav', 
                       src=waveform.cpu(), 
                       sample_rate=sample_rate)
        output_audio.seek(0)
        return base64.b64encode(output_audio.read()).decode('utf-8')
    finally:
        if output_audio is not None:
            output_audio.close()

def resample_for_model(audio: torch.Tensor, orig_sr: int, model_sr: int = 32000) -> torch.Tensor:
    """Resample audio to model's sample rate if needed."""
    if orig_sr == model_sr:
        return audio
    
    device = get_device()
    with resource_cleanup():
        resampler = T.Resample(orig_freq=orig_sr, new_freq=model_sr).to(device)
        resampled = resampler(audio)
        return resampled

def get_model_description(model_name: str, custom_description: Optional[str] = None) -> Optional[str]:
    """Get appropriate description based on model name or custom input."""
    if custom_description:
        return custom_description
    
    model_descriptions = {
        'thepatch/gary_orchestra': "violin, epic, film, piano, strings, orchestra",
        'thepatch/gary_orchestra_2': "violin, epic, film, piano, strings, orchestra"
    }
    return model_descriptions.get(model_name)

def _process_audio_impl(
    input_data_base64: str,
    model_name: str,
    config: AudioConfig,
    progress_callback: Optional[Callable] = None,
    description: Optional[str] = None
) -> str:
    model = None
    try:
        # Load and validate input
        song, sr = load_and_validate_audio(input_data_base64)
        
        # Resample input to model's sample rate if needed
        song_resampled = resample_for_model(song, sr)
        
        # Preprocess and wrap audio
        processed_waveform = preprocess_audio(song_resampled)
        wrapped_waveform = wrap_audio_if_needed(
            processed_waveform, 
            config.target_sr,  # Use model's sample rate for wrapping
            config.prompt_duration + config.output_duration
        )
        prompt_waveform = wrapped_waveform[..., :int(config.prompt_duration * config.target_sr)]
        
        # Initialize model
        with resource_cleanup():
            model = MusicGen.get_pretrained(model_name)
            if progress_callback:
                model.set_custom_progress_callback(progress_callback)
            
            model.set_generation_params(
                use_sampling=True,
                top_k=config.top_k,
                top_p=0.0,
                temperature=config.temperature,
                duration=config.output_duration,
                cfg_coef=config.cfg_coef
            )
            
            # Get description
            final_description = get_model_description(model_name, description)
            
            # Generate audio - IMPORTANT: Use config.target_sr here instead of sr
            print(f"Generating continuation with description: {final_description}")
            output = model.generate_continuation(
                prompt_waveform,
                prompt_sample_rate=config.target_sr,  # Use model's sample rate here
                descriptions=[final_description] if final_description else None,
                progress=True
            )
            
            if output is None or output.size(0) == 0:
                raise AudioProcessingError("Generated output is empty")
            
            # If original sample rate was different, resample back
            if sr != config.target_sr:
                output = resample_for_model(output, config.target_sr, sr)
            
            # Save output with original sample rate
            return save_audio_to_base64(output.squeeze(0), sr)
                
    except Exception as e:
        error_msg = f"Audio processing failed: {str(e)}"
        print(error_msg)
        raise AudioProcessingError(error_msg) from e
        
    finally:
        if model is not None:
            del model
        with resource_cleanup():
            pass

def _continue_music_impl(
    input_data_base64: str,
    model_name: str,
    config: AudioConfig,
    progress_callback: Optional[Callable] = None,
    description: Optional[str] = None
) -> str:
    model = None
    try:
        # Load and validate input
        song, sr = load_and_validate_audio(input_data_base64)
        
        # Ensure stereo
        if song.size(0) == 1:
            song = song.repeat(2, 1)
        
        # Resample for model if needed
        song_resampled = resample_for_model(song, sr)
        
        # Use the resampled audio for the model, with correct sample rate for duration calculation
        prompt_waveform = song_resampled[..., -int(config.prompt_duration * config.target_sr):]
        
        with resource_cleanup():
            # Initialize model
            model = MusicGen.get_pretrained(model_name)
            if progress_callback:
                model.set_custom_progress_callback(progress_callback)
            
            model.set_generation_params(
                use_sampling=True,
                top_k=config.top_k,
                top_p=0.0,
                temperature=config.temperature,
                duration=config.output_duration,
                cfg_coef=config.cfg_coef
            )
            
            # Get description
            final_description = get_model_description(model_name, description)
            
            print(f"Generating continuation with description: {final_description}")
            output = model.generate_continuation(
                prompt_waveform,
                prompt_sample_rate=config.target_sr,  # Use model's sample rate here
                descriptions=[final_description] if final_description else None,
                progress=True
            )
            
            # Process output
            output = output.squeeze(0) if output.dim() == 3 else output
            
            # If original sample rate was different, resample output back
            if sr != config.target_sr:
                output = resample_for_model(output, config.target_sr, sr)
            
            # Match channels with original audio
            original_minus_prompt = song[..., :-int(config.prompt_duration * sr)]
            if original_minus_prompt.size(0) != output.size(0):
                output = output.repeat(original_minus_prompt.size(0), 1)
            
            # Combine audio using original sample rate
            combined_waveform = torch.cat([original_minus_prompt, output], dim=1).to(get_device())
            
            # Save output with original sample rate
            return save_audio_to_base64(combined_waveform, sr)
                
    except Exception as e:
        error_msg = f"Music continuation failed: {str(e)}"
        print(error_msg)
        raise AudioProcessingError(error_msg) from e
        
    finally:
        if model is not None:
            del model
        with resource_cleanup():
            pass

# Backwards compatibility wrappers
def process_audio(
    input_data_base64: str,
    model_name: str,
    progress_callback: Optional[Callable] = None,
    prompt_duration: int = 6,
    top_k: int = 250,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    description: Optional[str] = None
) -> str:
    """
    Backwards-compatible wrapper for audio processing.
    """
    config = AudioConfig(
        prompt_duration=prompt_duration,
        top_k=top_k,
        temperature=temperature,
        cfg_coef=cfg_coef
    )
    return _process_audio_impl(
        input_data_base64,
        model_name,
        config,
        progress_callback,
        description
    )

def continue_music(
    input_data_base64: str,
    model_name: str,
    progress_callback: Optional[Callable] = None,
    prompt_duration: int = 6,
    top_k: int = 250,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    description: Optional[str] = None
) -> str:
    """
    Backwards-compatible wrapper for music continuation.
    """
    config = AudioConfig(
        prompt_duration=prompt_duration,
        top_k=top_k,
        temperature=temperature,
        cfg_coef=cfg_coef
    )
    return _continue_music_impl(
        input_data_base64,
        model_name,
        config,
        progress_callback,
        description
    )