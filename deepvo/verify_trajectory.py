import sys
import numpy as np
import matplotlib.pyplot as plt

seq_num = sys.argv[1]
kitti_path = '/mnt/hgfs/EECS568/kitti_dataset' # CHANGE THIS LINE ACCORDINGLY

gt = np.genfromtxt(kitti_path+'/poses/'+seq_num+'.txt')
abs_poses = np.genfromtxt('poses/'+seq_num+'_abs.txt')
gt = gt.reshape(-1, 3, 4)
abs_poses = abs_poses.reshape(-1, 3, 4)

gt_trans = gt[:,:,3]
abs_trans = abs_poses[:,:,3]
rmse = np.sqrt(np.mean((gt_trans-abs_trans)**2))

plt.plot(abs_poses[:,0,3], abs_poses[:,2,3], 'tab:blue', label='DeepVO Trajectory')
plt.plot(gt[:,0,3], gt[:,2,3], 'tab:green', label='Ground Truth')

print('RMSE Error: ', rmse)