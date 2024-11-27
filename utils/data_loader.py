import torch
from torch.utils.data import Dataset


class RandDataset(Dataset):
    def __init__(self, data, device='cpu'):
        """
        data: ndarray (m * n),
        """
        self.data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        self.len = data.shape[1]
        self.device = device

    def __getitem__(self, item):
        return self.data[:, item].to(device=self.device)

    def __len__(self):
        return self.len
