from torch.utils.data import Dataset
import torch


class HexDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.flatten(torch.tensor(y, dtype=torch.float32))
        return x, y

    def add_data(self, data, targets):
        self.data.append(data)
        self.targets.append(targets)
