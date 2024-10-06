import os
import json
import shutil

# Paths
output_dataset_path = "./dataset/gary"
train_jsonl_path = os.path.join(output_dataset_path, "train.jsonl")
test_jsonl_path = os.path.join(output_dataset_path, "test.jsonl")

# egs folder paths
egs_train_folder = "./egs/train"
egs_eval_folder = "./egs/eval"

# Function to load JSONL entries
def load_jsonl_entries(jsonl_path):
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            return [json.loads(line) for line in f]
    return []

# Function to save JSONL entries
def save_jsonl_entries(jsonl_path, entries):
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

# Function to filter entries
def filter_entries(entries):
    filtered_entries = []
    for entry in entries:
        if os.path.exists(entry["path"]):
            filtered_entries.append(entry)
        else:
            print(f"File {entry['path']} does not exist. Removing entry.")
    return filtered_entries

# Function to remove duplicate entries based on file path
def remove_duplicates(entries):
    seen_paths = set()
    unique_entries = []
    for entry in entries:
        if entry["path"] not in seen_paths:
            unique_entries.append(entry)
            seen_paths.add(entry["path"])
        else:
            print(f"Duplicate entry found and removed: {entry['path']}")
    return unique_entries

# Function to move and rename files, replacing if the file already exists
def move_and_rename_jsonl_file(src_path, dest_folder, new_filename):
    dest_path = os.path.join(dest_folder, new_filename)
    
    # Remove the destination file if it exists
    if os.path.exists(dest_path):
        os.remove(dest_path)
        print(f"Removed existing {dest_path}")
    
    # Copy the source file to the destination and rename it
    shutil.copy(src_path, dest_path)
    print(f"Moved and renamed {src_path} to {dest_path}")

# Load entries
train_entries = load_jsonl_entries(train_jsonl_path)
test_entries = load_jsonl_entries(test_jsonl_path)

# Filter entries
filtered_train_entries = filter_entries(train_entries)
filtered_test_entries = filter_entries(test_entries)

# Remove duplicates
unique_train_entries = remove_duplicates(filtered_train_entries)
unique_test_entries = remove_duplicates(filtered_test_entries)

# Save filtered and unique entries
save_jsonl_entries(train_jsonl_path, unique_train_entries)
save_jsonl_entries(test_jsonl_path, unique_test_entries)

# Move and rename 'train.jsonl' to 'egs/train/data.jsonl'
move_and_rename_jsonl_file(train_jsonl_path, egs_train_folder, "data.jsonl")

# Move and rename 'test.jsonl' to 'egs/eval/data.jsonl'
move_and_rename_jsonl_file(test_jsonl_path, egs_eval_folder, "data.jsonl")

print("Finished cleaning, deduplicating, and moving JSONL files.")
