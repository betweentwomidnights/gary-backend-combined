import os
import json
import subprocess
import random
import re
from pathlib import Path
from pydub import AudioSegment
import librosa
import numpy as np
from tqdm import tqdm
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import shutil
import shlex

from essentia_labels import genre_labels, mood_theme_classes, instrument_classes

# Paths
raw_audio_path = "./dataset/gary"
demucs_output_path = "./dataset/gary/demucs/htdemucs"
instrumental_output_path = "./dataset/gary/instrumental"
split_output_path = "./dataset/gary/split"
output_dataset_path = "./dataset/gary"
train_jsonl_path = os.path.join(output_dataset_path, "train.jsonl")
test_jsonl_path = os.path.join(output_dataset_path, "test.jsonl")

# Ensure the split folder exists
os.makedirs(split_output_path, exist_ok=True)
os.makedirs(instrumental_output_path, exist_ok=True)
os.makedirs(demucs_output_path, exist_ok=True)

# Demucs configuration
model = "htdemucs"
extensions = ["mp3", "wav", "ogg", "flac"]
two_stems = None
mp3 = True
mp3_rate = 320
float32 = False
int24 = False

# Function to find files with specific extensions in a path
def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out

# Function to check if a file has already been chunked
def is_file_chunked(raw_filename, split_dir):
    # Remove file extension from the raw audio filename
    raw_base_filename = os.path.splitext(raw_filename)[0]
    
    # List all chunked files in the split directory
    chunked_files = [f for f in os.listdir(split_dir) if f.startswith(raw_base_filename)]
    
    # If any chunked files exist with the same base name, consider the file chunked
    return len(chunked_files) > 0

# Function to separate audio using Demucs
def separate(inp, outp):
    cmd = ["demucs", "-n", model, "--two-stems=vocals", "--mp3", f"--mp3-bitrate={mp3_rate}", "--segment", "4", "-o", shlex.quote(str(outp))]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    files = [shlex.quote(str(f)) for f in find_files(inp)]
    if not files:
        print(f"No valid audio files in {inp}")
        return
    print("Going to separate the files:")
    print('\n'.join(files))
    print("With command: ", " ".join(cmd + files))
    p = subprocess.Popen(cmd + files, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print("Demucs output:", stdout.decode())
    print("Demucs errors:", stderr.decode())
    if p.returncode != 0:
        print("Command failed, something went wrong.")

# Function to slice and resample audio
def slice_and_resample_audio(dataset_path, output_dir, chunk_length=30000):
    print(f"Slicing and resampling audio in {dataset_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(dataset_path):
        if filename.endswith(".mp3"):
            if 'Zone.Identifier' in filename:
                continue
            audio_path = os.path.join(dataset_path, filename)
            if os.path.exists(audio_path):
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(44100)
                duration = len(audio)
                
                for i in range(0, duration - chunk_length, chunk_length):
                    chunk = audio[i:i + chunk_length]
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i//1000}.wav"
                    chunk.export(os.path.join(output_dir, chunk_filename), format="wav")
                
                if duration > chunk_length:
                    last_chunk = audio[-chunk_length:]
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{(duration - chunk_length)//1000}.wav"
                    last_chunk.export(os.path.join(output_dir, chunk_filename), format="wav")
                else:
                    last_chunk = audio
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk0.wav"
                    last_chunk.export(os.path.join(output_dir, chunk_filename), format="wav")

                print(f"Processed {filename}")
            else:
                print(f"File {audio_path} not found")
    print("All audio files processed.")

# Function to process Demucs output
def process_demucs_output(demucs_output_path, instrumental_output_path):
    print("Processing Demucs output...")
    os.makedirs(instrumental_output_path, exist_ok=True)

    for folder in os.listdir(demucs_output_path):
        first_level_folder_path = os.path.join(demucs_output_path, folder)
        if os.path.isdir(first_level_folder_path):
            model_folder_path = os.path.join(first_level_folder_path, "htdemucs")
            if os.path.isdir(model_folder_path):
                instrumental_file = os.path.join(model_folder_path, folder, "no_vocals.mp3")
                if os.path.exists(instrumental_file):
                    output_filename = f"{folder}.mp3"
                    new_path = os.path.join(instrumental_output_path, output_filename)
                    shutil.move(instrumental_file, new_path)
                    print(f"Moved and renamed {instrumental_file} to {new_path}")
                else:
                    print(f"No instrumental file found in {model_folder_path}/{folder}")
            else:
                print(f"Model directory {model_folder_path} does not exist")
        else:
            print(f"{first_level_folder_path} is not a directory")

# Function to rename files to remove spaces
def rename_files_to_remove_spaces(directory):
    for filename in os.listdir(directory):
        if ' ' in filename:
            new_filename = filename.replace(' ', '_')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed {filename} to {new_filename}")

# Function to process the raw dataset
def process_dataset(raw_audio_path, demucs_output_path, instrumental_output_path, split_output_path):
    rename_files_to_remove_spaces(raw_audio_path)
    for file in os.listdir(raw_audio_path):
        if file.endswith((".mp3", ".wav", ".flac", ".m4a")):
            input_file = os.path.join(raw_audio_path, file)
            output_folder = os.path.splitext(file)[0]
            cmd = [
                "demucs",
                "-n", "htdemucs",
                "--two-stems=vocals",
                "--mp3",
                "--mp3-bitrate", "320",
                "--segment", "4",
                "-o", shlex.quote(os.path.join(demucs_output_path, output_folder)),
                shlex.quote(input_file)
            ]
            print("Running command: ", " ".join(cmd))
            subprocess.run(" ".join(cmd), shell=True, check=True)
    
    process_demucs_output(demucs_output_path, instrumental_output_path)
    slice_and_resample_audio(instrumental_output_path, split_output_path)

# Function to filter predictions based on threshold
def filter_predictions(predictions, class_list, threshold=0.1):
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
    filtered_labels = [class_list[i] for i in filtered_indices]
    filtered_values = [predictions_mean[i] for i in filtered_indices]
    return filtered_labels, filtered_values

# Function to create comma-separated unique tags
def make_comma_separated_unique(tags):
    seen_tags = set()
    result = []
    for tag in ', '.join(tags).split(', '):
        if tag not in seen_tags:
            result.append(tag)
            seen_tags.add(tag)
    return ', '.join(result)

# Function to extract the artist name from the filename
def extract_artist_from_filename(filename):
    match = re.search(r'(.+?)\s\d+_chunk\d+\.wav', filename)
    artist = match.group(1) if match else ""
    return artist.replace("mix", "").strip() if "mix" in artist else artist

# Function to get audio features
def get_audio_features(audio_filename):
    audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    result_dict = {}

    genre_model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    predictions = genre_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, genre_labels)
    filtered_labels = ', '.join(filtered_labels).replace("---", ", ").split(', ')
    result_dict['genres'] = make_comma_separated_unique(filtered_labels)

    mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    predictions = mood_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, mood_theme_classes, threshold=0.05)
    result_dict['moods'] = make_comma_separated_unique(filtered_labels)

    instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
    predictions = instrument_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, instrument_classes)
    result_dict['instruments'] = filtered_labels

    return result_dict

# Function to append to JSONL files
def append_to_jsonl(jsonl_path, new_entries):
    with open(jsonl_path, "a") as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

# Function to autolabel and create split datasets
def autolabel_and_create_split(split_dataset_path, output_dataset_path):
    train_entries = []
    eval_entries = []

    with open(os.path.join(output_dataset_path, "train.jsonl"), "a") as train_file, \
         open(os.path.join(output_dataset_path, "test.jsonl"), "a") as eval_file:
        dset = os.listdir(split_dataset_path)
        random.shuffle(dset)
        for filename in tqdm(dset):
            if 'Zone.Identifier' in filename:
                continue
            try:
                result = get_audio_features(os.path.join(split_dataset_path, filename))
            except:
                result = {"genres": [], "moods": [], "instruments": []}
            y, sr = librosa.load(os.path.join(split_dataset_path, filename))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = round(tempo[0]) if isinstance(tempo, np.ndarray) else round(tempo)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key = np.argmax(np.sum(chroma, axis=1))
            key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
            length = librosa.get_duration(y=y, sr=sr)
            artist_name = extract_artist_from_filename(filename)
            entry = {
                "key": f"{key}",
                "artist": artist_name,
                "sample_rate": 44100,
                "file_extension": "wav",
                "description": "",
                "keywords": "",
                "duration": length,
                "bpm": tempo,
                "genre": result.get('genres', ""),
                "title": filename,
                "name": "",
                "instrument": result.get('instruments', ""),
                "moods": result.get('moods', []),
                "path": os.path.join(split_dataset_path, filename)
            }
            if random.random() < 0.85:
                train_entries.append(entry)
            else:
                eval_entries.append(entry)

    append_to_jsonl(train_jsonl_path, train_entries)
    append_to_jsonl(test_jsonl_path, eval_entries)

# Main function
if __name__ == "__main__":
    # Check for new audio files that haven't been chunked yet
    new_files_to_process = []
    for file in os.listdir(raw_audio_path):
        if file.endswith((".mp3", ".wav", ".flac", ".m4a")):
            if not is_file_chunked(file, split_output_path):
                new_files_to_process.append(file)
    
    if new_files_to_process:
        print(f"New files to process: {new_files_to_process}")
        vocals_remove = input("Do you need to remove vocals? (y/n): ").strip().lower()
        if vocals_remove == 'y':
            process_dataset(raw_audio_path, demucs_output_path, instrumental_output_path, split_output_path)
        
        split_and_resample = input("Do you need to split and resample audio? (y/n): ").strip().lower()
        if split_and_resample == 'y':
            slice_and_resample_audio(instrumental_output_path, split_output_path)
        
        autolabel = input("Do you need to autolabel and update JSONL files? (y/n): ").strip().lower()
        if autolabel == 'y':
            autolabel_and_create_split(split_output_path, output_dataset_path)
    else:
        print("No new files to process.")
    
    print("Process completed.")
