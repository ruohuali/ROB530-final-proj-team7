# CHANGES NEEDED TO RUN
# deepvo/params.py: change self.data_dir to your KITTI dataset path
# deepvo/model/: download the link below and place in this folder
# https://drive.google.com/file/d/1l0s3rYWgN8bL0Fyofee8IhN-0knxJF22/view

import gtsam
import sys
from deepvo.deepvo_handler import DeepVOHandler
from deepvo.helper import R_to_angle, eulerAnglesToRotationMatrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

seq_num = sys.argv[1]

dvoh = DeepVOHandler(seq_num)
N = dvoh.get_len()
N = min(N, 100)

# Ground truth for plotting
gt = np.genfromtxt('/mnt/hgfs/EECS568/kitti_dataset/poses/'+seq_num+'.txt')
gt = gt.reshape(gt.shape[0], 3, 4)

# Intialize ISAM2
params = gtsam.ISAM2Params()
params.setRelinearizeThreshold(0.1)
params.setRelinearizeSkip(1)
isam = gtsam.ISAM2(params)

# Use same noise everywhere (can change this)
noise = gtsam.noiseModel.Diagonal.Sigmas([0.01,0.01,0.01,0.01,0.01,0.01])

# Intialize graph and prior
graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), noise))

# Use relative pose for first iteration
rel_pose, _ = dvoh.get_pose(0, [0,0,0,0,0,0])
abs_pose = rel_pose
R = gtsam.Rot3(eulerAnglesToRotationMatrix(rel_pose[:3]))
t = rel_pose[3:]
pose = gtsam.Pose3(R, t)
initial.insert(0, pose)

# Track poses
vo_pose, slam_pose = [], []
vo_pose.append(pose.translation())
slam_pose.append(pose.translation())

loop = False
cur_est = initial
for i in range(1, N):
    print(str(i)+' / '+str(N), end='\r', flush=True)

    # Use previously optimized vertex as prev_pose
    pose = cur_est.atPose3(i-1)
    initial.insert(i, pose)
    prev_pose = np.zeros((3,4))
    prev_pose[:,:3] = pose.rotation().matrix()
    prev_pose[:,3] = pose.translation()
    prev_pose = R_to_angle(prev_pose)[:6]
    
    # Grab transformation from DeepVO
    #rel_pose, _ = dvoh.get_pose(i, prev_pose)
    rel_pose, _ = dvoh.get_pose(i, abs_pose)
    _, abs_pose = dvoh.get_pose(i, abs_pose)

    # Define edge using relative pose between frames
    quat = Rotation.from_matrix(eulerAnglesToRotationMatrix(rel_pose[:3])).as_quat()
    R = gtsam.Rot3.Quaternion(quat[3], quat[0], quat[1], quat[2])
    #R = gtsam.Rot3(eulerAnglesToRotationMatrix(rel_pose[:3]))
    ang = eulerAnglesToRotationMatrix([0, prev_pose[0], 0])
    t = ang.dot(rel_pose[3:])
    #t = rel_pose[3:]
    edge = gtsam.Pose3(R, t)
    
    # loop = checkForLoopClosure() @ i
    if not loop:
        factor = gtsam.BetweenFactorPose3(i-1, i, edge, noise)
    else:
        # Need to change None to the img_idx of frame we're closing the loop with
        factor = gtsam.BetweenFactorPose3(i-1, None, edge, noise)
    graph.add(factor)
    
    isam.update(graph, initial)
    cur_est = isam.calculateEstimate()
    
    vo_pose.append(abs_pose[3:])
    slam_pose.append(cur_est.atPose3(i).translation())
    initial.clear()
    
vo_pose = np.array(vo_pose)
slam_pose = np.array(slam_pose)

plt.plot(vo_pose[:,0], vo_pose[:,2], 'tab:blue', label='DeepVO Trajectory')
plt.plot(slam_pose[:,0], slam_pose[:,2], 'tab:orange', label='Optimized Trajectory')
plt.plot(gt[:N,0,3], gt[:N,2,3], 'tab:green', label='Ground Truth')
plt.legend()
plt.axis('equal')
plt.show()

