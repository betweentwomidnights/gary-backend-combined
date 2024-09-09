
# g4lwebsockets

this is the backend for the max4live device.

`docker-compose-g4lwebsockets.yml` is what we use to spin up the whole shebang. (renamed to `docker-compose.yml` unless someone knows how to do this better)

## main scripts

- `Dockerfile.g4lwebsockets`: thecollabagepatch/g4lwebsockets:latest
- `entrypoint.sh`
- `requirements-g4lwebsockets.txt`

- `g4lwebsockets.py` (using gevent greenlets with flask_socket-io)

    - we use mongoDB and redis docker images. at one point we also just ran them locally.
    - we use gridFS so that mongoDB and redis don't get mad at how big our audio files get.

- `g4laudio.py` (the continuation functions)

there are 3 functions that can be called by the device on the front-end. **'bang'**, **'continue'**, and **'retry'**.

### 'bang'

is the initial generation, which uses the process_audio_request.

in `g4lwebsockets.py`, it's this function:

```python
(line 96) 
@socketio.on('process_audio_request') 
def handle_audio_processing(data):
```

which calls this function in `g4laudio.py`:

```python
(line 38) 
def process_audio(input_data_base64, model_name, progress_callback=None, prompt_duration=6):
```

in the `process_audio` function, **'prompt_duration'** is taken from the beginning of the sample. The **'wrap_audio_if_needed'** (line 31 of `g4laudio.py`) function was intended to prevent errors if the input_audio is shorter than the prompt_duration, but our front-end will always send 30 second samples. it adds silence to the end of the input_data_base64 (in that repo `myBuffer.wav`).

**`process_audio`** always sends back 30 seconds of base64 audio data.

### 'continue'

uses the **continue_music** request.

in `g4lwebsockets.py`, it's this function:

```python
(line 139)
@socketio.on('continue_music_request')
def handle_continue_music(data):
```

which calls this function in `g4laudio.py`:

```python
(line 101) 
def continue_music(input_data_base64, musicgen_model, progress_callback=None, prompt_duration=6):
```

in the `continue_music` function, the prompt_duration is taken from the end of the input_data_base64, and then combined with the input_audio. It also always produces 30 seconds of base64 audio data, but its outputs can be longer than that due to the concatenation with the entire input_data_base64.

### 'retry'

using 'retry' in the max4live device triggers the **retry_music_request**.

in `g4lwebsockets.py`:

```python
(line 184)
@socketio.on('retry_music_request')
def handle_retry_music(data):
```

this function actually just uses the **continue_music** function in `g4laudio.py` again on the exact same input_data_base64 that was last used. Using the session, we keep track so that we can have the model go again from the exact same place with different parameters, or with the same parameters if we want.

---

the max4live device has a **crop** function using `ffmpeg`, as musicgen often ends its outputs with abrupt silence. the user can crop the audio and update the session with the new input_data_base64 so that they can use the **continue_music_request** function without having the silence in the prompt_duration.

in `g4lwebsockets.py`:

```python
@socketio.on('update_cropped_audio')
def handle_update_cropped_audio(data):
```

---

Finally, there's a standard flask route apart from the websockets that has a supposedly robust health check we can call. we use it at https://thecollabagepatch.com to update everyone on the status of the backend.

in `g4lwebsockets.py`:

```python
(line 243) 
@app.route('/health', methods=['GET'])
```

---

### some notes about stuff:

in `g4laudio.py`, we have two currently unused functions. dnb gary's a noisy boy so we experimented with preprocessing the input audio for him and decided it wasn't doing a whole lot.

```python
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
```

we use a **custom_progress_callback** to send the progress of generation to the frontend.

in `g4laudio.py`:

```python
(line 66 and 115)
model_continue.set_custom_progress_callback(progress_callback)
```

it's this function in `audiocraft/models/musicgen.py`:

```python
def set_custom_progress_callback(self, progress_callback):
    def internal_progress_callback(current_step, total_steps):
        if current_step % 50 == 0:  # Emit progress every 50 steps
            if progress_callback:
                progress_callback(current_step, total_steps)
    
    self._progress_callback = internal_progress_callback
```
