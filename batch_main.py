import gtsam
import sys
import numpy as np
import matplotlib.pyplot as plt

def toGTSAMEdge(vo_pose):
    R = gtsam.Rot3(vo_pose[:,:3])
    t = gtsam.Point3(vo_pose[:,3])
    gtsam_pose = gtsam.Pose3(R, t)
    return gtsam_pose

seq_num = sys.argv[1]
kitti_path = '/mnt/hgfs/EECS568/kitti_dataset' # CHANGE THIS LINE ACCORDINGLY

# Ground truth from KITTI & relative and absolute poses from DeepVO
gt = np.genfromtxt(kitti_path+'/poses/'+seq_num+'.txt')
rel_poses = np.genfromtxt('deepvo/poses/'+seq_num+'_rel.txt')
abs_poses = np.genfromtxt('deepvo/poses/'+seq_num+'_abs.txt')

# Reshape to (N,3,4)
N = gt.shape[0]
gt = gt.reshape(N, 3, 4)
rel_poses = rel_poses.reshape(N, 3, 4)
abs_poses = abs_poses.reshape(N, 3, 4)

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
noise = gtsam.noiseModel.Diagonal.Sigmas([0.01,0.01,0.01,0.01,0.01,0.01])
graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), noise))
initial.insert(0, gtsam.Pose3())

for i in range(1,N):
    vertex = toGTSAMEdge(abs_poses[i])
    initial.insert(i, vertex)

    edge = toGTSAMEdge(rel_poses[i])
    factor = gtsam.BetweenFactorPose3(i-1, i, edge, noise)
    graph.add(factor)

params = gtsam.GaussNewtonParams()
params.setRelativeErrorTol(1e-5)
params.setMaxIterations(100)
optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
result = optimizer.optimize()

optm_poses = np.zeros((N,3))
for i in range(N):
    optm_poses[i] = result.atPose3(i).translation()

plt.plot(abs_poses[:N,0,3], abs_poses[:N,2,3], 'tab:blue', label='DeepVO Trajectory')
plt.plot(optm_poses[:N,0], optm_poses[:N,2], 'tab:orange', label='Optimized Trajectory')
plt.plot(gt[:N,0,3], gt[:N,2,3], 'tab:green', label='Ground Truth')
plt.legend()
plt.axis('equal')
plt.show()
