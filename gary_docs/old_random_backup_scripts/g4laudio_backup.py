import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import base64
import io
import uuid
import torchaudio.transforms as T
import gc

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

# Function to normalize audio to a target peak amplitude
def peak_normalize(y, target_peak=0.9):
    return target_peak * (y / np.max(np.abs(y)))

# Function to normalize audio to a target RMS value
def rms_normalize(y, target_rms=0.05):
    current_rms = np.sqrt(np.mean(y**2))
    return y * (target_rms / current_rms)

def preprocess_audio(waveform):
    waveform_np = waveform.cpu().squeeze().numpy()  # Move tensor to CPU and convert to numpy
    processed_waveform_np = waveform_np  # Use the waveform as-is without processing
    return torch.from_numpy(processed_waveform_np).unsqueeze(0).cuda()  # Convert back to tensor and move to GPU

# Function to wrap audio if needed
def wrap_audio_if_needed(waveform, sr, desired_duration):
    current_duration = waveform.shape[-1] / sr
    while current_duration < desired_duration:
        waveform = torch.cat([waveform, waveform[:, :int((desired_duration - current_duration) * sr)]], dim=-1)
        current_duration = waveform.shape[-1] / sr
    return waveform

def process_audio(input_data_base64, model_name, progress_callback=None, prompt_duration=6):
    # Decode the base64 input data
    input_data = base64.b64decode(input_data_base64)
    input_audio = io.BytesIO(input_data)

    try:
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
            del resampler  # Ensure resampler is deleted
            torch.cuda.empty_cache()
        else:
            song_resampled = song

        # Preprocess the resampled audio
        processed_waveform = preprocess_audio(song_resampled)

        # Load the model
        model_continue = MusicGen.get_pretrained(model_name)
        model_continue.set_custom_progress_callback(progress_callback)

        # Setting generation parameters
        output_duration = song_resampled.shape[-1] / expected_sr
        model_continue.set_generation_params(
            use_sampling=True,
            top_k=250,
            top_p=0.0,
            temperature=1.0,
            duration=output_duration,
            cfg_coef=3.0,
        )

        # Ensure the input waveform is long enough
        prompt_waveform = wrap_audio_if_needed(processed_waveform, expected_sr, prompt_duration + output_duration)
        prompt_waveform = prompt_waveform[..., :int(prompt_duration * expected_sr)]

        # Conditionally apply the description for specific models
        if model_name in ['thepatch/gary_orchestra', 'thepatch/gary_orchestra_2']:
            # Convert the description into a single string
            description = "soundtrack, violin, film, piano"
        else:
            # Default to None if not using the specific models
            description = None

        
        # Generate the continuation with dynamic description
        output = model_continue.generate_continuation(
            prompt_waveform, 
            prompt_sample_rate=expected_sr, 
            descriptions=[description] if description else None,  # Pass the description if available
            progress=True
        )

        # Convert output tensor to a compatible format before saving
        output = output.float()  # Ensure dtype is torch.float32
        # Convert the output tensor to a byte buffer
        output_audio = io.BytesIO()
        torchaudio.save(output_audio, format='wav', src=output.cpu().squeeze(0), sample_rate=expected_sr)
        output_audio.seek(0)
        output_data_base64 = base64.b64encode(output_audio.read()).decode('utf-8')

    finally:
        # Cleanup to ensure all resources are freed
        input_audio.close()
        del model_continue, song, song_resampled, processed_waveform, prompt_waveform, output, input_data, output_audio
        torch.cuda.empty_cache()
        gc.collect()

    return output_data_base64

def continue_music(input_data_base64, musicgen_model, progress_callback=None, prompt_duration=6):
    # Decode the base64 input data
    input_data = base64.b64decode(input_data_base64)
    input_audio = io.BytesIO(input_data)

    try:
        song, sr = torchaudio.load(input_audio)
        song = song.to('cuda')  # Assume CUDA is available and preferred

        # Normalize audio channels
        if song.size(0) == 1:
            song = song.repeat(2, 1)  # Make stereo if mono

        model_continue = MusicGen.get_pretrained(musicgen_model)
        model_continue.set_custom_progress_callback(progress_callback)
        model_continue.set_generation_params(
            use_sampling=True,
            top_k=250,
            top_p=0.0,
            temperature=1.0,
            duration=30,  # Set duration explicitly
            cfg_coef=3.0
        )

        # Generate continuation
        prompt_waveform = song[:, -int(prompt_duration * sr):]

        # Conditionally apply the description for specific models
        if musicgen_model in ['thepatch/gary_orchestra', 'thepatch/gary_orchestra_2']:
            # Convert the description into a single string
            description = "soundtrack, violin, film, piano"
        else:
            # Default to None if not using the specific models
            description = None

        
        # Generate the continuation with dynamic description
        output = model_continue.generate_continuation(
            prompt_waveform, 
            prompt_sample_rate=sr, 
            descriptions=[description] if description else None,  # Pass the description if available
            progress=True
        )
        output = output.squeeze(0) if output.dim() == 3 else output  # Ensure 2D tensor for audio

        # Resample output if necessary
        if sr != 32000:
            resampler = T.Resample(orig_freq=32000, new_freq=sr).to('cuda')
            output = resampler(output)
            del resampler  # Ensure resampler is deleted
            torch.cuda.empty_cache()

        # Ensure all tensors are on the same device and have the same number of channels
        original_minus_prompt = song[:, :-int(prompt_duration * sr)]
        if original_minus_prompt.size(0) != output.size(0):
            # Adjust channel numbers if needed
            output = output.repeat(original_minus_prompt.size(0), 1)

        # Concatenate tensors
        combined_waveform = torch.cat([original_minus_prompt, output], dim=1).to('cuda')


        # Convert output tensor to a compatible format before saving
        output = output.float()  # Ensure dtype is torch.float32

        # Save output
        output_audio = io.BytesIO()
        torchaudio.save(output_audio, format='wav', src=combined_waveform.cpu(), sample_rate=sr)
        output_audio.seek(0)
        output_data_base64 = base64.b64encode(output_audio.read()).decode('utf-8')

    finally:
        # Cleanup to ensure all resources are freed
        input_audio.close()
        del song, prompt_waveform, output, combined_waveform, model_continue, output_audio, input_data
        torch.cuda.empty_cache()
        gc.collect()

    return output_data_base64