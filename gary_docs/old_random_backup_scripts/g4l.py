import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import sys

def peak_normalize(y, target_peak=0.9):
    """Normalize the audio to a target peak amplitude."""
    return target_peak * (y / np.max(np.abs(y)))

def rms_normalize(y, target_rms=0.05):
    """Normalize the audio to a target RMS value."""
    current_rms = np.sqrt(np.mean(y**2))
    return y * (target_rms / current_rms)

def preprocess_audio(waveform):
    waveform_np = waveform.squeeze().numpy()  # Convert tensor to numpy array
    processed_waveform_np = rms_normalize(peak_normalize(waveform_np))
    return torch.from_numpy(processed_waveform_np).unsqueeze(0)  # Convert back to tensor

def wrap_audio_if_needed(waveform, sr, desired_duration):
    # Calculate the current duration of the waveform
    current_duration = waveform.shape[-1] / sr

    # If the waveform's duration is shorter than the desired duration, loop it until it's long enough
    while current_duration < desired_duration:
        waveform = torch.cat([waveform, waveform[:, :int((desired_duration - current_duration) * sr)]], dim=-1)
        current_duration = waveform.shape[-1] / sr

    return waveform

def main():
    # Hardcoded paths
    input_audio_path = "C:/gary4live/g4l/myBuffer.wav"
    output_audio_path = "C:/gary4live/g4l/myOutput"  # Remove the .wav extension here
    
    desired_sr = 24000  # The sample rate MusicGen expects

    # Load the input audio
    song, sr = torchaudio.load(input_audio_path)

    # Resample the audio to the desired sample rate
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sr)
    song_resampled = resampler(song)
    
    # Define the desired prompt duration (in seconds)
    prompt_duration = 6.0
    output_duration = song_resampled.shape[-1] / desired_sr

    # Preprocess the audio
    processed_waveform = preprocess_audio(song_resampled)

    # Hardcoded base path
    base_path = "C:/gary4live/audiocraft/models/"

    # Check if a model name is provided as an argument
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'facebook/musicgen-small'

    # Define a list of default models that shouldn't have the base path applied
    default_models = [
        'facebook/musicgen-small',
        'facebook/musicgen-medium',
        'facebook/musicgen-melody',
        'facebook/musicgen-large'
    ]

    # Complete the model path
    if model_name in default_models:
        model_path = model_name  # Don't apply base path for default models
    else:
        model_path = base_path + model_name  # Apply base path for custom models

    # Load the model
    model_continue = MusicGen.get_pretrained(model_path)



    # Setting generation parameters
    model_continue.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        duration=output_duration,
        cfg_coef=3.0
    )

    # Ensure the input waveform is long enough
    prompt_waveform = wrap_audio_if_needed(processed_waveform, desired_sr, prompt_duration + output_duration)
    prompt_waveform = prompt_waveform[..., :int(prompt_duration * desired_sr)]  # Extract only the prompt duration

    # Generate the continuation
    output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=desired_sr, progress=True)

    # If you want to save the output at the original sample rate:
    resampler_back = torchaudio.transforms.Resample(orig_freq=desired_sr, new_freq=sr)
    output_resampled = resampler_back(output.squeeze(0).cpu())

    # Save the output audio
    audio_write(output_audio_path, output.squeeze(0).cpu(), model_continue.sample_rate, strategy="loudness", loudness_compressor=True)


if __name__ == "__main__":
    main()
