# g4laudio.py

import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import base64
import io

# Function to normalize audio to a target peak amplitude
def peak_normalize(y, target_peak=0.9):
    return target_peak * (y / np.max(np.abs(y)))

# Function to normalize audio to a target RMS value
def rms_normalize(y, target_rms=0.05):
    current_rms = np.sqrt(np.mean(y**2))
    return y * (target_rms / current_rms)

def preprocess_audio(waveform):
    waveform_np = waveform.cpu().squeeze().numpy()  # Move tensor to CPU and convert to numpy
    # Skip normalization and RMS processing
    # processed_waveform_np = rms_normalize(peak_normalize(waveform_np))
    processed_waveform_np = waveform_np  # Use the waveform as-is without processing
    return torch.from_numpy(processed_waveform_np).unsqueeze(0).cuda()  # Convert back to tensor and move to GPU

# Function to wrap audio if needed
def wrap_audio_if_needed(waveform, sr, desired_duration):
    current_duration = waveform.shape[-1] / sr
    while current_duration < desired_duration:
        waveform = torch.cat([waveform, waveform[:, :int((desired_duration - current_duration) * sr)]], dim=-1)
        current_duration = waveform.shape[-1] / sr
    return waveform

def process_audio(input_data_base64, model_name):
    # Decode the base64 input data
    input_data = base64.b64decode(input_data_base64)
    input_audio = io.BytesIO(input_data)

    # Load the input audio
    song, sr = torchaudio.load(input_audio)
    song = song.cuda()  # Move the tensor to GPU for processing

    # Model's expected sample rate
    expected_sr = 32000  # Adjust this value based on your model's requirements

    # Check if the input audio's sample rate matches the model's expected sample rate
    if sr != expected_sr:
        # Resample the audio to match the model's expected sample rate
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=expected_sr).cuda()
        song_resampled = resampler(song)
    else:
        song_resampled = song

    # Preprocess the resampled audio
    processed_waveform = preprocess_audio(song_resampled)

    # Load the model
    model_continue = MusicGen.get_pretrained(model_name)

    # Setting generation parameters
    output_duration = song_resampled.shape[-1] / expected_sr
    model_continue.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        duration=output_duration,
        cfg_coef=3.0
    )

    prompt_duration = 6.0  # The desired prompt duration in seconds
    # Ensure the input waveform is long enough
    prompt_waveform = wrap_audio_if_needed(processed_waveform, expected_sr, prompt_duration + output_duration)
    prompt_waveform = prompt_waveform[..., :int(prompt_duration * expected_sr)]

    # Generate the continuation
    output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=expected_sr, progress=True)

    # Convert the output tensor to a byte buffer
    output_audio = io.BytesIO()
    torchaudio.save(output_audio, format='wav', src=output.cpu().squeeze(0), sample_rate=expected_sr)
    output_audio.seek(0)
    output_data_base64 = base64.b64encode(output_audio.read()).decode('utf-8')

    return output_data_base64

def continue_music(input_data_base64, musicgen_model):
    # Decode the base64 input data
    input_data = base64.b64decode(input_data_base64)
    input_audio = io.BytesIO(input_data)

    # Load the input audio ensuring consistent channel layout and format
    song, sr = torchaudio.load(input_audio)
    song = song.cuda()  # Move the tensor to GPU for processing

    # Ensure the audio is stereo if the original audio is mono
    if song.size(0) == 1:
        song = song.repeat(2, 1)  # Make stereo if mono

    # Load the model and set generation parameters
    model_continue = MusicGen.get_pretrained(musicgen_model)
    model_continue.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        duration=30,  # This is the output duration for the continuation part
        cfg_coef=3.0
    )

    prompt_duration = 6.0  # The duration of audio to use for generating continuation

    # Slice the last 'prompt_duration' seconds of audio to use as a prompt
    prompt_waveform = song[:, -int(prompt_duration * sr):]

    # Generate the continuation
    output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)

    # Remove the extra batch dimension from output if necessary
    if output.dim() == 3:
        output = output.squeeze(0)  # Remove batch dimension if exists

    # Ensure the output has the same number of channels as the original
    if output.size(0) != song.size(0):
        output = output.repeat(song.size(0), 1)  # Match the channel count to the original

    # Debug: Check shapes
    print(f"Original shape: {song.shape}, Output shape: {output.shape}")

    # Combine the original audio minus the prompt with the generated continuation
    original_minus_prompt = song[:, :-int(prompt_duration * sr)]
    combined_waveform = torch.cat([original_minus_prompt, output], dim=1)  # Concatenate along the time axis

    # Convert the combined tensor to a byte buffer
    output_audio = io.BytesIO()
    torchaudio.save(output_audio, format='wav', src=combined_waveform.cpu(), sample_rate=sr)
    output_audio.seek(0)
    output_data_base64 = base64.b64encode(output_audio.read()).decode('utf-8')

    return output_data_base64