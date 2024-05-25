import blobfile as bf
import numpy as np


def list_files(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        results.append(full_path)
    return results


def normalize(image, MIN_BOUND=None, MAX_BOUND=None, max_val=1.0):
    if MIN_BOUND is None:
        MIN_BOUND = image.min()
    if MAX_BOUND is None:
        MAX_BOUND = image.max()
    eps = 1e-8
    image = (image - MIN_BOUND + eps) / (MAX_BOUND - MIN_BOUND + eps) * max_val
    return image


def list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "dcm", 'gz']:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results
