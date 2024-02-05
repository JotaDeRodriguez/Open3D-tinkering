import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

DATANAME = ("ITC_groundfloor.ply")
pcd = o3d.io.read_point_cloud(r'C:/Users/JuanRodriguez/Documents/LPS/PointClouds/' + DATANAME)

voxel_size = 0.025
pcd_downsampled = pcd.voxel_down_sample(voxel_size = voxel_size)

nn = 600
std_multiplier = 5

filtered_pcd, filtered_idx = pcd_downsampled.remove_statistical_outlier(nn, std_multiplier)

print(pcd_downsampled)

pcd_downsampled.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=128))

assert (pcd_downsampled.has_normals())

# using all defaults
oboxes = pcd_downsampled.detect_planar_patches(
    normal_variance_threshold_deg=60,
    coplanarity_deg=75,
    outlier_ratio=0.5,
    min_plane_edge_length= 0.2,
    min_num_points=500,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))

print("Detected {} patches".format(len(oboxes)))

geometries = []
for obox in oboxes:
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
    mesh.paint_uniform_color(obox.color)
    geometries.append(mesh)
    geometries.append(obox)
geometries.append(pcd)

print(geometries)
o3d.visualization.draw_geometries(geometries)
