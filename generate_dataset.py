"""
Converts a dataset in LJSpeech format into audio tokens that can be used to train/fine-tune Soprano.
This script creates two JSON files for train and test splits in the provided directory.

Usage:
python generate_dataset.py --input-dir path/to/files

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
"""
import argparse
import pathlib
import random
import json

from scipy.io import wavfile
import torchaudio
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from encoder.codec import Encoder

# Constants
SAMPLE_RATE = 32000
SEED = 42
VAL_PROP = 0.1
VAL_MAX = 512

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = args.input_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model onto {device}.")
    encoder = Encoder().to(device)
    encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    print("Model loaded.")

    print("Reading metadata.")
    files = []
    metadata_path = input_dir / 'metadata.txt'
    with open(metadata_path, encoding='utf-8') as f:
        data = f.read().strip().split('\n')
        for line in data:
            if '|' not in line:
                continue
            filename, transcript = line.split('|', maxsplit=1)
            files.append((filename, transcript))
    print(f'{len(files)} samples located in directory.')

    print(f"Encoding audio on {device}...")
    dataset = []
    for filename, transcript in tqdm(files):
        wav_path = input_dir / 'wavs' / f'{filename}.wav'
        
        try:
            sr, audio = wavfile.read(wav_path)
        except FileNotFoundError:
            continue
            
        # Convert to float tensor for torchaudio/encoder compatibility
        audio = torch.from_numpy(audio).float()
        
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        
        # Prepare for encoder
        audio = audio.unsqueeze(0).to(device)
        with torch.no_grad():
            audio_tokens = encoder(audio)
        
        # Store results (move back to CPU for JSON serialization)
        dataset.append([transcript, audio_tokens.squeeze(0).cpu().tolist()])

    print("Generating train/test splits.")
    random.seed(SEED)
    random.shuffle(dataset)
    num_val = min(int(VAL_PROP * len(dataset)) + 1, VAL_MAX)
    train_dataset = dataset[num_val:]
    val_dataset = dataset[:num_val]
    print(f'# train samples: {len(train_dataset)}')
    print(f'# val samples: {len(val_dataset)}')

    print("Saving datasets.")
    with open(input_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    with open(input_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
    print("Datasets saved.")

if __name__ == '__main__':
    main()
