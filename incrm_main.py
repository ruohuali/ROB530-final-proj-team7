import gtsam
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def mat2Euler(mat):
    e_ang = Rotation.from_matrix(mat).as_euler('xyz', degrees=False)
    return [0, e_ang[1], 0]

def euler2Mat(e_ang):
    mat = Rotation.from_euler('xyz', e_ang, degrees=False).as_matrix()
    return mat

def toGTSAMEdgeOnlyYaw(vo_pose):
    e_ang = mat2Euler(vo_pose[:,:3])
    mat = euler2Mat(e_ang)
    R = gtsam.Rot3(mat)
    t = gtsam.Point3(vo_pose[:,3])
    gtsam_pose = gtsam.Pose3(R, t)
    return gtsam_pose

seq_num = sys.argv[1]
kitti_path = '/mnt/hgfs/EECS568/kitti_dataset' # CHANGE THIS LINE ACCORDINGLY

# Ground truth from KITTI & relative and absolute poses from DeepVO
gt = np.genfromtxt(kitti_path+'/poses/'+seq_num+'.txt')
rel_poses = np.genfromtxt('deepvo/poses/'+seq_num+'_rel.txt')
abs_poses = np.genfromtxt('deepvo/poses/'+seq_num+'_abs.txt')
try:
    loops = np.genfromtxt('resnet_loop_closure/gt_loops/'+seq_num+'_loops.txt')
except:
    loops = np.zeros((0,1))

# Reshape to (N,3,4)
N = gt.shape[0]
gt = gt.reshape(N, 3, 4)
rel_poses = rel_poses.reshape(N, 3, 4)
abs_poses = abs_poses.reshape(N, 3, 4)
#N = min(N, 300) # COMMENT OUT THIS LINE TO RUN FOR FULL SEQUENCE

# Intialize ISAM2
params = gtsam.ISAM2Params()
params.setRelinearizeThreshold(0.1)
params.setRelinearizeSkip(1)
isam = gtsam.ISAM2(params)

# Use same noise everywhere (can change this)
odom_noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6)/10)
loop_noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6))

# Intialize graph and prior
graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), odom_noise))
initial.insert(0, gtsam.Pose3())

optm_poses = np.zeros((N,3))
cur_est = initial
for i in range(1,N):
    print(str(i)+' / '+str(N), end='\r', flush=True)

    prev_pose = cur_est.atPose3(i-1)
    initial.insert(i, prev_pose)

    edge = toGTSAMEdgeOnlyYaw(rel_poses[i])
    factor = gtsam.BetweenFactorPose3(i-1, i, edge, odom_noise)
    graph.add(factor)

    # Attemp at loop closure in gtsam
    #loop_detect = np.argwhere(loops[:,0] == i)
    #if len(loop_detect) > 0:
    #    loop = loop_detect[0][0]
    #    factor = gtsam.BetweenFactorPose3(i, loop, gtsam.Pose3(), loop_noise)
    #    graph.add(factor)

    isam.update(graph, initial)
    cur_est = isam.calculateEstimate()

    optm_poses[i] = cur_est.atPose3(i).translation()
    initial.clear()

plt.plot(abs_poses[:N,0,3], abs_poses[:N,2,3], 'tab:blue', label='DeepVO Trajectory')
plt.plot(optm_poses[:N,0], optm_poses[:N,2], 'tab:orange', label='Optimized Trajectory')
plt.plot(gt[:N,0,3], gt[:N,2,3], 'tab:green', label='Ground Truth')
plt.legend()
plt.axis('equal')
plt.show()