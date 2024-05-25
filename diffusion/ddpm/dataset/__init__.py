import blobfile as bf
from torch.utils.data import DataLoader
from .ultra_dataset import UltraDataset
from .vs_dataset import MRIDataset
from .ct_dataset import CTDataset


def load_dataset(
    data_dir,
    batch_size,
    image_size,
    data_type,
):
    if not data_dir:
        raise ValueError("unspecified datasets directory")
    data_factory = {
        'ultra': UltraDataset,
        'mri': MRIDataset,
        'ct': CTDataset,
    }
    datafunc = data_factory[data_type]
    dataset = datafunc(data_dir, image_size)
    print(f"load {len(dataset)} {data_type} samples")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    return loader
