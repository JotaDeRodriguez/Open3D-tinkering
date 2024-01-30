import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

DATANAME = "ITC_groundfloor.ply"
pcd = o3d.io.read_point_cloud(r'C:/Users/JuanRodriguez/Documents/LPS/PointClouds/' + DATANAME)

pcd_center = pcd.get_center()
pcd.translate(-pcd_center)


voxel_size = 0.1
pcd_downsampled = pcd.voxel_down_sample(voxel_size = voxel_size)



print(pcd)
print(pcd_downsampled)

nn_distance = np.mean(pcd_downsampled.compute_nearest_neighbor_distance())

segment_models={}
segments={}

max_plane_idx=5

epsilon = 0.1
min_cluster_points = 5

rest=pcd_downsampled
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold=0.1,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    labels = np.array(segments[i].cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]

best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])

rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
segments[i]=segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))

colors = plt.get_cmap("tab20")(i)
segments[i].paint_uniform_color(list(colors[:3]))

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)])
