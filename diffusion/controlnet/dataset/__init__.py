from re import split
from .ultra_datasets import UltraDataset
from .vs_datasets import MRIDataset
from torch.utils.data import DataLoader
from .ct_datasets import CTDataset


def load_dataset(
    data_dir,
    ann_dir,
    data_type,
    batch_size,
    image_size,
):
    if not data_dir:
        raise ValueError("unspecified datasets directory")
    data_factory = {
        'ultra': UltraDataset,
        'mri': MRIDataset,
        'ct': CTDataset,
    }
    datafunc = data_factory[data_type]
    
    train_dataset = datafunc(
        data_dir,
        ann_dir,
        image_size,
        split='train',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    test_dataset = datafunc(
            data_dir,
            ann_dir,
            image_size,
            split='test'
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    return train_loader, test_loader
