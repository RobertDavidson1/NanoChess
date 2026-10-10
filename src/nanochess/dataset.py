import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


def collate(batch):
    # zip each tuple (x,y) into x's and y's
    x_s, y_s = zip(*batch)

    x_s = pad_sequence(
        x_s,
        batch_first=True,
        padding_value=1 # Pad with <EOS>
    )

    y_s = pad_sequence(
        y_s,
        batch_first=True,
        padding_value=-100
    )


    # Return batch padded to max length in batch
    return x_s, y_s

class ChessDataset(Dataset):
    def __init__(self, offset_path, token_path):
        self.tokens = np.memmap(
            token_path, dtype=np.uint16, mode="r"
        )
        self.offsets = np.memmap(
            offset_path, dtype=np.uint64, mode="r"
        )

    def __len__(self):
        return len(self.offsets)


    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        
        # Get the tokens for example[i] 
        # Convert to long to store the `-100` pad ID value
        tokens = torch.tensor(self.tokens[start:end], dtype = torch.long)


        # Target is next token x_i+1 given x_i
        x = tokens[:-1]
        y = tokens[1:]

        return x,y


def get_loader(split = "train", batch_size = 8):
    if not split in ["train",  "val"]:
        raise ValueError("split must be 'train' or 'val'")
    
    
    data = Path("data/processed")
    offset_path = data / "train_offsets.bin" if split == "train" else data / "val_offsets.bin"
    token_path = data/ "train.bin" if split == "train" else data / "val.bin"


    dataset = ChessDataset(offset_path, token_path)
    return DataLoader(dataset, batch_size, True, collate_fn=collate)