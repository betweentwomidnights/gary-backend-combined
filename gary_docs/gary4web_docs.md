
# gary4web

this backend is for the web app (gary4web) and the chrome-extension (gary-on-the-fly).

this one also has an **express-server** for the ffmpeg cropping to make web front-ends more lightweight. It makes the `docker-compose.yml` a little more complicated, but it's not so bad.

## main scripts

- `Dockerfile.concurrent_gary`: thecollabagepatch/concurrent_gary:latest
- `Dockerfile.redis`
- `express-server/Dockerfile`: thecollabagepatch/express-server:latest
- `server.js`
- `cron`: (this file is to restart the backend every 12 hours due to the gpu cache slowly filling up for inexplicable reasons. i might have actually solved this problem)
- `concurrent_gary.py` (using flask with gunicorn and an rq worker)
    - we use the standard mongoDB docker image and our custom redis docker image.
- `g4laudio.py` (the continue_music function in here is shared between both backends)
- `requirements-concurrent_gary.txt`

## important routes

### 1. **/generate**

```python
(line 242)
@app.route('/generate', methods=['POST'])
def generate_audio():
    data = request.json
    youtube_url = data['url']
    timestamp = data.get('currentTime', 0)
    model = data.get('model', 'facebook/musicgen-small')
    promptLength = int(data.get('promptLength', 6))
    duration = data.get('duration', '16-18').split('-')
```

this route performs the initial generation based upon the input audio (with the timestamp of the playback cursor being the start of the generation).

the input audio here is downloaded using **yt-dlp** on the youtube url sent from the front-end.

this route calls this function:

```python
(line 177)
def process_youtube_url(youtube_url, timestamp, model, promptLength, min_duration, max_duration, task_id):
```

which calls this function:

```python
(line 159)
def generate_audio_continuation(prompt_waveform, sr, bpm, model, min_duration, max_duration, progress_callback=None):
```

a note about the bpm detection: this script was the first backend we made when transitioning from the standalone python script 'gary.py' which attempted to automate the process of stitching together outputs into a seamless arrangement. we left it here because it's still nice when an output doesn't need to be cropped.

this route accepts a 'duration range' in order to try and perform this bpm detection to make an output stop at the end of a bar. the default is '16-18'. musicgen can go up to 30.

---

### 2. **/continue**

this route is for extending the already existing generations in the web-app/chrome-extension.

```python
(line 291)
@app.route('/continue', methods=['POST'])
def continue_audio():
    data = request.json
    task_id = data['task_id']
    musicgen_model = data['model']
    prompt_duration = int(data.get('prompt_duration', 6))
    input_data_base64 = data['audio']  # Get the audio data from the request
```

this route calls this function in `concurrent_gary.py`:

```python
(line 210)
def process_continuation(task_id, input_data_base64, musicgen_model, prompt_duration):
```

which in turn uses **continue_music** 

in `g4laudio.py`:

```python
(line 101)
def continue_music(input_data_base64, musicgen_model, progress_callback=None, prompt_duration=6):
```

this route always produces 30 seconds of audio, then it combines it with the input audio.

---

### other routes

the other routes in `concurrent_gary.py` are designed to keep track of tasks using the rq worker for the user to be able to queue up multiple generations concurrently.

#### **/tasks**

this route was the first attempt to track task id for enqueuing jobs.

```python
(line 324)
@app.route('/tasks/<jobId>', methods=['GET'])
def get_task(jobId):
    try:
        task = audio_tasks.find_one({'_id': ObjectId(jobId)})
        if task:
            return Response(json.dumps(task, default=json_util.default), mimetype='application/json')
        else:
            return jsonify({"error": "Task not found"}), 404
    except bson.errors.InvalidId:
        return jsonify({"error": "Invalid ObjectId format"}), 400
```

#### **/fetch-result**

```python
@app.route('/fetch-result/<taskId>', methods=['GET'])
def fetch_result(taskId):
    try:
        task = audio_tasks.find_one({'_id': ObjectId(taskId)})
        if task:
            if task.get('status') == 'completed':
                return jsonify({"status": "completed", "audio": task.get('audio')})
            else:
                return jsonify({"status": task.get('status')})
        else:
            return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

/fetch-result was supposed to aid in this.

#### **/progress**

this route uses the custom callback of 50 steps.

```python
(line 349)
@app.route('/progress/<taskId>', methods=['GET'])
def get_progress(taskId):
    try:
        progress = redis_conn.get(f"progress_{taskId}")
        if progress:
            return jsonify({"progress": float(progress)})
        else:
            return jsonify({"progress": 0.0})  # Default to 0 if no progress found
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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

---

### **/health**

we can use this route at our website to inform users of backend status.

```python
(line 360)
@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "mongodb": "down",
        "pytorch": "down",
        "redis": "down",
        "status": "down"
    }
```

---

## some notes about stuff:

i'm not super happy with the rq worker. it seems to really slow down generations and we don't have enough users to rly need a job queue like this. i also tried to do 3 rq workers to see how scalable this could be, and i couldn't get it working. if i remember right, it has to do with the pytorch operations and cool words like pickling.

i think alot of this backend could be rewritten using something faster than mongoDB/redis/rq worker.