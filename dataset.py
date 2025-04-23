import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class PokemonDataset(Dataset):
    def __init__(self, root_dir="data/train", transform=None):
        self.root_dir = root_dir

        assert os.path.exists(
            self.root_dir
        ), f"Root directory {self.root_dir} does not exist."

        # glob match all images in the directory
        self.images = [
            os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)
        ]

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
