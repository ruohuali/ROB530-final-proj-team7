import sys
import numpy as np
from deepvo_handler import DeepVOHandler
from helper import eulerAnglesToRotationMatrix

seq_num = sys.argv[1]

dvoh = DeepVOHandler(seq_num)
N = dvoh.get_len()

rel_save, abs_save = np.zeros((N,3,4)), np.zeros((N,3,4))
abs_pose = [0,0,0,0,0,0]

for i in range(1, N):
    print(str(i)+' / '+str(N), end='\r', flush=True)

    rel_pose, abs_pose = dvoh.get_pose(i, abs_pose)

    rel_save[i,:,:3] =  eulerAnglesToRotationMatrix(rel_pose[:3])
    rel_save[i,:,3] = rel_pose[3:]
    abs_save[i,:,:3] =  eulerAnglesToRotationMatrix(abs_pose[:3])
    abs_save[i,:,3] = abs_pose[3:]

np.savetxt('poses/'+seq_num+'_rel.txt', rel_save.reshape(N,-1))
np.savetxt('poses/'+seq_num+'_abs.txt', abs_save.reshape(N,-1))
