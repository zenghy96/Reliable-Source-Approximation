import blobfile as bf
import numpy as np


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def list_files(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        results.append(full_path)
    return results


def norm_img(image):
    MAX_BOUND = image.max()
    MIN_BOUND = image.min()
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) * 255
    return np.uint8(image)
