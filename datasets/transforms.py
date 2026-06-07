import numpy as np


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    radius = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if radius > 0:
        pc = pc / radius
    return pc


def random_sample(pc, num):
    permutation = np.arange(np.size(pc, 0))
    np.random.shuffle(permutation)
    return pc[permutation[:num]]


def default_pc_transform(pc):
    pc = random_sample(pc, min(10000, np.size(pc, 0)))
    return pc_normalize(pc)
