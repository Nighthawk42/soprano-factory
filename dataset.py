import json
import pathlib
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path):
        # Convert string path to Path object if necessary for consistency
        self.path = pathlib.Path(path)
        with open(self.path, encoding='utf-8') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, audio = self.dataset[idx]
        
        # Format: [STOP][TEXT]<text prompt>[START]<audio tokens>[STOP]
        # Optimization: Use a generator expression for joining tokens
        audio_tokens = ''.join(f'[{x}]' for x in audio)
        res = f"[STOP][TEXT]{text}[START]{audio_tokens}[STOP]"
        
        return res