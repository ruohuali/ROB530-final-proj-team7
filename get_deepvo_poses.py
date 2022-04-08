import sys
from deepvo.deepvo_handler import DeepVOHandler
from deepvo.helper import eulerAnglesToRotationMatrix
import numpy as np
import matplotlib.pyplot as plt

seq_num = sys.argv[1]


dvoh = DeepVOHandler(seq_num)
N = dvoh.get_len()

# Ground truth for plotting
gt = np.genfromtxt('/Users/lounisbouzit/Documents/EECS568/kitti_dataset/poses/'+seq_num+'.txt')
gt = gt.reshape(gt.shape[0], 3, 4)
print(N, gt.shape[0])

rel_save, abs_save = np.zeros((N,3,4)), np.zeros((N,3,4))
abs_pose = [0,0,0,0,0,0]

for i in range(1, N):
    print(str(i)+' / '+str(N), end='\r', flush=True)

    rel_pose, abs_pose = dvoh.get_pose(i, abs_pose)
    
    rel_save[i,:,:3] =  eulerAnglesToRotationMatrix(rel_pose[:3])
    rel_save[i,:,3] = rel_pose[3:]
    abs_save[i,:,:3] =  eulerAnglesToRotationMatrix(abs_pose[:3])
    abs_save[i,:,3] = abs_pose[3:]

np.savetxt('deepvo/poses/'+seq_num+'_rel.txt', rel_save.reshape(N,-1))
np.savetxt('deepvo/poses/'+seq_num+'_abs.txt', abs_save.reshape(N,-1))

plt.plot(abs_save[:,0,3], abs_save[:,2,3], 'tab:blue', label='DeepVO')
plt.plot(gt[:,0,3], gt[:,2,3], 'tab:green', label='Ground Truth')
plt.legend()
plt.axis('equal')
plt.show()