import os
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt


class MRIDataset(Dataset):
    def __init__(self, data_dir, ann_dir, resolution, split):
        super(MRIDataset, self).__init__()
        subjects_name = os.listdir(data_dir)
        anns_name = os.listdir(ann_dir)
        subjects_name = sorted(subjects_name, key=lambda x: int(x.split('_')[-1]))
        anns_name = sorted(anns_name, key=lambda x: int(x.split('_')[-1]))
        images_0, images_1, masks_0, masks_1, _0_ids, _1_ids = [], [], [], [], [], []
        if split == 'train':
            n_images = 8000
            # subjects_name = subjects_name[100:150]
            # anns_name = anns_name[100:150]
        else:
            n_images = 100
            subjects_name = subjects_name[200:]
            anns_name = anns_name[200:]
        self.split = split
        for subject_name, ann_name in zip(subjects_name, anns_name):
            assert subject_name == ann_name
            data = sitk.ReadImage(f"{data_dir}/{subject_name}/vs_gk_t1_refT1.nii.gz")
            data = sitk.GetArrayFromImage(data).astype(np.float32)
            ann = sitk.ReadImage(f"{data_dir}/{ann_name}/vs_gk_seg_refT1.nii.gz")
            ann = sitk.GetArrayFromImage(ann).astype(np.uint8)
            for i, (img, mask) in enumerate(zip(data, ann)):
                img = normalize(img)
                img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_AREA)
                if mask.max() > 0:
                    images_1.append(img)
                    masks_1.append(mask)
                    _1_ids.append(f"{subject_name}-{i}")
                else:
                    images_0.append(img)
                    masks_0.append(mask)
                    _0_ids.append(f"{subject_name}-{i}")  

        n_0 = int(np.median([0, len(images_0), n_images - len(images_1)]))
        self.images = images_1 + images_0[:n_0]
        self.masks = masks_1 + masks_0[:n_0]
        self.imgs_id = _1_ids + _0_ids[:n_0]
        print(f'load {len(images_1)} images with VS, {n_0} without VS')

        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5]),
        ])
        self.condition_transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        img_id = self.imgs_id[index]

        if random.random() < 0.4 and self.split == 'train':
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        condition = generate_condition(img, mask)
        # condition = generate_random_condition(img)
        condition = self.condition_transform(condition)
        img = self.image_transforms(img)
        return img, condition, img_id


# def generate_condition(image, mask):
#     threshs = [60, 80, 100, 120]
#     image = (image * 255).astype(np.uint8)
#     if mask.max() == 0:
#         final_T = random.choice(threshs)
#     else:
#         best_edge_num = float('inf')
#         best_T = 90
#         for T in threshs:
#             edge = generate_edge(image, T)
#             inter_num, edge_num = match_mask_edge(mask, edge)
#             if inter_num > 0 and edge_num < best_edge_num:
#                 best_edge_num = edge_num
#                 best_T = T
#         # final_T = random.uniform(best_T-10, best_T+10)
#         final_T = best_T
#     condition = generate_edge(image, 70)
#     return condition


def generate_condition(image, mask):
    threshs = [120, 110, 100, 90]
    image = (image * 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0)
    # if mask.max() == 0:
    #     edge = cv2.Canny(image, 110, 110)
    # else:
    #     for T in threshs:
    #         edge = cv2.Canny(image, T, T)
    #         if is_good_edge(edge, mask):
    #             break
    edge = cv2.Canny(image, 115, 115)
    condition = edge
    return condition


def is_good_edge(edge, mask):
    # count mask edge pixel number
    mask_edge = cv2.Canny(mask, 1, 1)
    _, mask_edge = cv2.threshold(mask_edge, 0, 1, cv2.THRESH_BINARY)
    num = mask_edge.sum()
    # count inter edge pixel number
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask1 = cv2.dilate(mask, kernel, iterations=1)
    mask2 = cv2.erode(mask, kernel, iterations=1)
    mask_ = mask1 - mask2
    inter = mask_ * edge
    inter_num = np.sum(inter>0)
    return inter_num >= num * 0.7


def match_mask_edge(mask, edge):
    # count mask edge pixel number
    mask_edge = cv2.Canny(mask, 1, 1)
    _, mask_edge = cv2.threshold(mask_edge, 0, 1, cv2.THRESH_BINARY)
    num = mask_edge.sum()
    # count inter edge pixel number
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask1 = cv2.dilate(mask, kernel, iterations=1)
    mask2 = cv2.erode(mask, kernel, iterations=1)
    mask_ = mask1 - mask2
    inter = mask_ * edge
    plt.imshow(inter)
    # count edge pixel number
    inter_num = np.sum(inter>0)
    if inter_num < num * 0.9:
        edge_num = float('inf')
    else:
        edge_num = edge.sum()
    return inter_num, edge_num


def generate_edge(image, T):
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=3)
    edge = cv2.Canny(image, T, T+5)
    return edge


def normalize(image):
    min_bound = image.min()
    max_bound = image.max()
    image = (image - min_bound) / (max_bound - min_bound) * 1.0
    return image


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from utils import _list_files
    data_dir = "data/Vestibular-Schwannoma-SEG"
    save_dir = "outputs/VS_T2"
    os.makedirs(save_dir, exist_ok=True)
    ann_dir = data_dir
    train_dataset = MRIDataset(
        data_dir,
        ann_dir,
        320,
        split='train'
    )
    for i in range(len(train_dataset)):
        # i=35
        img, condition, img_id = train_dataset[i]
        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(img[0])
        plt.subplot(122)
        plt.imshow(condition[0])
        plt.savefig(f'{save_dir}/{img_id}.png')
        plt.close()