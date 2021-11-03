from torchvision import transforms
from torch.utils.data import Dataset


class Dataloader(Dataset):
    def __init__(self, ):
        self.toTensor = transforms.ToTensor()
        x = 0

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return 0

    # def _



if __name__ == '__main__':
    dataloader = Dataloader()