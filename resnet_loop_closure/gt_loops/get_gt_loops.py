import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

kitti_path = '/mnt/hgfs/EECS568/kitti_dataset' # CHANGE THIS LINE ACCORDINGLY
plot = True

for seq_num in ['00','01','02','03','04','05','06','07','08','09','10']:
    print('seq_num: ', seq_num)
    gt = np.genfromtxt(kitti_path+'/poses/'+seq_num+'.txt')
    N = gt.shape[0]
    gt = gt.reshape(N, 3, 4)

    loops = []
    for i in range(100,N):
        prev_poses = gt[:i-99,:,3]
        cur_pose = gt[i,:,3].reshape(1,-1)

        dist = np.linalg.norm(cur_pose - prev_poses, axis=1)
        min_idx = np.argmin(dist)
        #print(dist[min_idx])
        if dist[min_idx] < 5: # must be within 0.75 meters
            loops.append([i, min_idx])

    if not len(loops):
        print('No loops detected\n')
        continue
    loops = np.array(loops)
    np.savetxt(seq_num+'_loops.txt', loops)

    if plot:
        plt.clf()
        steps = N//10
        print(N, steps)
        for j in range(0,N,steps):
            plt.plot(gt[j:j+steps,0,3], gt[j:j+steps,2,3], c=np.random.rand(3,))
        for loop in loops:
            plt.scatter(gt[loop[1],0,3], gt[loop[1],2,3], marker='s', color='b')
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.show()
    print(' ')

