from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
from . import _list_files

class ImageDataset(Dataset):
    def __init__(self, data_dir, resolution):
        super(ImageDataset, self).__init__()
        self.images_path = _list_files(data_dir)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        return image

