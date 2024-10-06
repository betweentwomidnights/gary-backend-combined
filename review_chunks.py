import os
import json
import subprocess
from pathlib import Path
import re

# Paths
split_dataset_path = "./dataset/gary/split"
output_dataset_path = "./dataset/gary"
train_jsonl_path = os.path.join(output_dataset_path, "train.jsonl")
test_jsonl_path = os.path.join(output_dataset_path, "test.jsonl")

# Function to play audio using ffplay
def play_audio(file_path):
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", file_path], check=True)
    except Exception as e:
        print(f"Could not play {file_path}: {e}")

# Function to remove file
def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Could not delete {file_path}: {e}")

# Load existing JSONL entries
def load_jsonl_entries(jsonl_path):
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            return [json.loads(line) for line in f]
    return []

train_entries = load_jsonl_entries(train_jsonl_path)
test_entries = load_jsonl_entries(test_jsonl_path)

# Map file paths to entries
def map_entries_by_path(entries):
    return {entry["path"]: entry for entry in entries}

train_entries_by_path = map_entries_by_path(train_entries)
test_entries_by_path = map_entries_by_path(test_entries)

# Extract artist from filename
def extract_artist_from_filename(filename):
    match = re.search(r'(.+?)\s\d+_chunk\d+\.wav', filename)
    artist = match.group(1) if match else ""
    return artist.replace("mix", "").strip() if "mix" in artist else artist

# Iterate through files in the split directory
for filename in os.listdir(split_dataset_path):
    file_path = os.path.join(split_dataset_path, filename)
    if filename.endswith(".wav"):
        play_audio(file_path)
        user_input = input(f"Keep {filename}? (y/n): ").strip().lower()
        if user_input == 'n':
            remove_file(file_path)
            # Remove from JSONL entries
            if file_path in train_entries_by_path:
                train_entries_by_path.pop(file_path)
                print(f"Removed {file_path} from train entries")
            if file_path in test_entries_by_path:
                test_entries_by_path.pop(file_path)
                print(f"Removed {file_path} from test entries")
        elif user_input == 'y':
            print(f"Kept {filename}")

# Write updated entries back to JSONL files
with open(train_jsonl_path, "w") as train_file, open(test_jsonl_path, "w") as eval_file:
    for entry in train_entries_by_path.values():
        train_file.write(json.dumps(entry) + '\n')
    for entry in test_entries_by_path.values():
        eval_file.write(json.dumps(entry) + '\n')

print("Finished processing audio chunks.")
