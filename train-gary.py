import os
import json
import subprocess
import logging
from audiocraft.utils import export
from audiocraft import train

# Set up logging
logging.basicConfig(level=logging.INFO)

def setup_directories(folder_dataset_is_saved_in, audiocraft_path):
    train_path = os.path.join(audiocraft_path, "egs/train")
    eval_path = os.path.join(audiocraft_path, "egs/eval")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    # Copy train and test JSONL files to the appropriate directories
    subprocess.run(["cp", os.path.join(folder_dataset_is_saved_in, "train.jsonl"), train_path], check=True)
    subprocess.run(["cp", os.path.join(folder_dataset_is_saved_in, "test.jsonl"), eval_path], check=True)
    subprocess.run(["cp", os.path.join(folder_dataset_is_saved_in, "train.yaml"), os.path.join(audiocraft_path, "config/dset/audio/train.yaml")], check=True)

    # Create data.jsonl files
    create_data_jsonl(os.path.join(train_path, "data.jsonl"), os.path.join(folder_dataset_is_saved_in, "train.jsonl"))
    create_data_jsonl(os.path.join(eval_path, "data.jsonl"), os.path.join(folder_dataset_is_saved_in, "test.jsonl"))

def create_data_jsonl(output_path, input_jsonl_path):
    with open(input_jsonl_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            f_out.write(line)

def run_training_command(num_epochs, folder_to_save_checkpoints_in):
    os.environ['USER'] = 'lyra'
    os.environ['DORA_DIR'] = '/tmp/audiocraft_lyra'

    command = (
        "dora -P audiocraft run "
        " solver=musicgen/musicgen_base_32khz"
        " model/lm/model_scale=small"
        " continue_from=//pretrained/facebook/musicgen-small"
        " conditioner=text2music"
        " dset=audio/train"
        " dataset.num_workers=1"
        " dataset.valid.num_samples=1"
        " dataset.batch_size=2"
        " schedule.cosine.warmup=6"
        " optim.optimizer=dadam"
        " optim.lr=1"  # Higher learning rate as used in the original Audiocraft paper
        f" optim.epochs={num_epochs}"
        " optim.updates_per_epoch=600"
        #" optim.weight_decay=0.01"
        " generate.lm.prompted_samples=False"
        " generate.lm.gen_gt_samples=True"
    )



    # Run the training command
    logging.info(f"Running training command: {command}")
    subprocess.run(command, shell=True, check=True)

def export_checkpoint(checkpoint_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    xp = train.main.get_xp_from_sig(checkpoint_folder)
    export.export_lm(xp.folder / 'checkpoint.th', os.path.join(output_dir, 'state_dict.bin'))
    export.export_pretrained_compression_model('facebook/encodec_32khz', os.path.join(output_dir, 'compression_state_dict.bin'))

if __name__ == "__main__":
    try:
        # Set paths and parameters
        folder_dataset_is_saved_in = "/workspace/dataset/gary"
        audiocraft_path = "/workspace/audiocraft"
        num_epochs = 4
        folder_to_save_checkpoints_in = "/workspace/checkpoints"
        
        # Setup directories and prepare data files
        setup_directories(folder_dataset_is_saved_in, audiocraft_path)
        
        # Run the training
        run_training_command(num_epochs, folder_to_save_checkpoints_in)
        
        # Export the checkpoint after training
        root_dir = "/tmp/audiocraft_lyra/xps/"
        subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        joined_paths = [os.path.join(root_dir, subfolder) for subfolder in subfolders]
        SIG = max(joined_paths, key=os.path.getmtime)
        
        export_checkpoint(SIG, folder_to_save_checkpoints_in)
        
        logging.info("Training and export completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")