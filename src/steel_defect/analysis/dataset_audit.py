"""Optional image integrity and similarity helpers, separate from inference."""
import hashlib
from collections import defaultdict
from PIL import Image

def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def exact_duplicates(paths):
    groups = defaultdict(list)
    for p in paths:
        groups[hash_file(p)].append(str(p))
    return {h: paths for h, paths in groups.items() if len(paths) > 1}

def compute_phash(path):
    import imagehash
    with Image.open(path) as image:
        return imagehash.phash(image.convert('L'))

def closest_phash_pairs(paths_a, paths_b, top_k=10):
    import heapq
    a, b = [(p, compute_phash(p)) for p in paths_a], [(p, compute_phash(p)) for p in paths_b]
    return heapq.nsmallest(top_k, ((ha-hb, str(pa), str(pb)) for pa, ha in a for pb, hb in b))

def load_small_image(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (64, 64))

def closest_ssim_pairs(paths_a, paths_b, top_k=10):
    import heapq
    from skimage.metrics import structural_similarity
    a, b = [(p, load_small_image(p)) for p in paths_a], [(p, load_small_image(p)) for p in paths_b]
    return heapq.nlargest(top_k, ((float(structural_similarity(ia, ib, data_range=255)), str(pa), str(pb))
                                    for pa, ia in a for pb, ib in b))
