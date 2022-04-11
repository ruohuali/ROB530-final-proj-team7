import sys
import numpy as np
import matplotlib.pyplot as plt

seq_num = sys.argv[1]
kitti_path = '/mnt/hgfs/EECS568/kitti_dataset' # CHANGE THIS LINE ACCORDINGLY

gt = np.genfromtxt(kitti_path+'/poses/'+seq_num+'.txt')
orb = np.genfromtxt('CameraTrajectory.txt')
gt = gt.reshape(-1, 3, 4)
orb = orb.reshape(-1, 3, 4)

plt.plot(orb[:,0,3], orb[:,2,3], 'tab:blue', label='ORB-SLAM2 + DeepVO')
plt.plot(gt[:,0,3], gt[:,2,3], 'tab:green', label='Ground Truth')
plt.title('Sequence '+seq_num)
plt.legend()
plt.axis('equal')
plt.show()