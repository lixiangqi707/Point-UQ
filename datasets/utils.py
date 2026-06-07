from random import sample

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData


def fix_off_header(file_path):
    with open(file_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    if not lines:
        raise ValueError(f"File {file_path} is empty.")

    parts = lines[0].strip().split()
    if len(parts) != 3 and parts[0] != "OFF":
        raise ValueError(f"Invalid OFF header in file {file_path}: {lines[0]}")

    if len(parts) == 3 and parts[0].startswith("OFF"):
        original_header = lines[0]
        lines[0] = "OFF\n"
        lines.insert(1, original_header[3:].strip() + "\n")
        with open(file_path, "w", encoding="utf-8") as fout:
            fout.writelines(lines)


def offread_uniformed(filepath, sampled_pt_num=10000):
    fix_off_header(filepath)
    mesh = o3d.io.read_triangle_mesh(filepath)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {filepath} has no triangles")
    point_cloud = mesh.sample_points_uniformly(sampled_pt_num)
    return np.asarray(point_cloud.points, dtype=np.float32)


def farthest_point_sampling(points, target_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.farthest_point_down_sample(num_samples=target_points)
    return np.asarray(downsampled.points, dtype=np.float32)


def plyread(filepath, target_points=1024):
    data = PlyData.read(filepath)
    vertex = data["vertex"]
    points = np.vstack([vertex[t].astype(np.float32) for t in ("x", "y", "z")]).T
    num_points = points.shape[0]

    if num_points == target_points:
        return points
    if num_points > target_points:
        return farthest_point_sampling(points, target_points)

    sampled_points = points.copy()
    extra_points_needed = target_points - num_points
    indices = list(range(num_points))
    while len(indices) < extra_points_needed:
        indices.extend(sample(range(num_points), min(extra_points_needed - len(indices), num_points)))
    extra_points = points[np.array(indices[:extra_points_needed])]
    return np.vstack((sampled_points, extra_points)).astype(np.float32)


def read_from_path(path: str):
    if path.endswith(".pt"):
        return torch.load(path)
    if path.endswith(".ply"):
        return np.asarray(plyread(path, target_points=1024), dtype=np.float32)
    if path.endswith(".off"):
        return offread_uniformed(path, sampled_pt_num=1024)
    if path.endswith(".npy"):
        points = np.load(path).astype(np.float32)
        if points.shape[0] > 1024:
            points = farthest_point_sampling(points, 1024)
        return points.astype(np.float32)
    raise ValueError(f"Unsupported file extension for path: {path}")
