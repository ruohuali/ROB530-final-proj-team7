# NOTES FOR RUNNING CODE
# deepvo/params.py: update self.data_dir to your KITTI datasets location
# deepvo/data_helper.py: change lines 23/24 and 242 accoridng to the KITTI dataset you're using

import socket
import struct
import numpy as np
import torch
import sys
from deepvo.params import par
from deepvo.model import DeepVO
from deepvo.data_helper import get_data_info, ImageSequenceDataset
from deepvo.helper import eulerAnglesToRotationMatrix, R_to_angle
import array

def floatList2Bytes(lst):
    buf = bytes()
    for val in lst:
        buf += struct.pack('d', val)
    return buf

def toSE3(x):
    pose = np.zeros((4,4))
    R = np.array(eulerAnglesToRotationMatrix(x[:3]))
    t = np.array(x[3:])
    pose[:3,:3] = R
    pose[:3,3] = t
    pose[3,3] = 1
    return pose

np.set_printoptions(suppress=True) # remove scientific notation when printing
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8080  # Port to listen on (non-privileged ports are > 1023)
seq_num = sys.argv[1]
seq_len, overlap, batch_size = 6, 5, 1


# Initialize and load pretrained model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
#if torch.cuda.is_available():
    # M_deepvo.to('cuda')
#    M_deepvo.load_state_dict(torch.load('deepvo/model/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train'))
#else:
M_deepvo.load_state_dict(torch.load('deepvo/model/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train', map_location={'cuda:0': 'cpu'}))
M_deepvo.eval()

# Preprocess all input images
df = get_data_info(folder_list=[seq_num], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
df = df.loc[df.seq_len == seq_len]  # drop last
dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
print('Finished loading dataset..\n')


# Set up socket to client
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))

while True:

    # Recieve message from ORB SLAM
    s.listen()
    conn, addr = s.accept()
    #print(f"Connected by {addr}")
    data = conn.recv(136)
    img_idx, recv_mTcw = 0, np.zeros(16, dtype=np.float64)
    for i in range(0, len(data), 8):
        if i == 0:
            img_idx = int(struct.unpack('<d', data[i:i+8])[0])
        else:
            recv_mTcw[i//8 - 1] = float(struct.unpack('<d', data[i:i+8])[0])
    recv_mTcw = recv_mTcw.reshape(4, 4)
    recv_mTwc = np.linalg.inv(recv_mTcw)
    pose_15 = R_to_angle(recv_mTwc[:3])
    prev_pose = pose_15[:6]

    print("Received message from client...")
    print('received idx:', img_idx)
    print('Previous pose recieved: ', prev_pose)
    if not data:
        break
    #*************************************
    # Set imageindex AND POSE FROM SERVER HERE
    # img_idx += 1
    # prev_pose = [0,0,0,0,0,0]
    #*************************************

    # Index differently in first sequence
    if img_idx < 5:
        ds_idx = 0
        pose_idx = img_idx
    else:
        ds_idx = img_idx-5
        pose_idx = -1

    # Pass through DeepVO to get relative pose for all seq in seq_len
    x = dataset[ds_idx][1]
    x = x[None,:] # mimic batch size of 1
    batch_predict_pose = M_deepvo.forward(x).data.cpu().numpy()
    
    # Only keep relevant idx in sequence
    predict_pose_seq = batch_predict_pose[0][pose_idx]

    # Transform x,y,z using yaw of vehicle
    ang = eulerAnglesToRotationMatrix([0, prev_pose[0], 0])
    location = ang.dot(predict_pose_seq[3:])
    predict_pose_seq[3:] = location

    # Add relative pose to absolute
    send_mTwc = [a + b for a, b in zip(predict_pose_seq, prev_pose)]
    send_mTwc[0] = (send_mTwc[0]+np.pi)%(2*np.pi)-np.pi # normalize to [-pi,pi] over y-axis
    send_mTwc = toSE3(send_mTwc)
    send_mTcw = np.linalg.inv(send_mTwc)
    print('Absolute pose at ', img_idx)
    print(send_mTcw)

    # Send pose to ORB SLAM
    abs_pose_lst_bytes = floatList2Bytes(send_mTcw.reshape(-1).tolist())
    #abs_pose_lst_bytes = floatList2Bytes(np.linalg.inv(abs_pose).reshape(-1).tolist())
    conn.sendall(abs_pose_lst_bytes)

    print('=================================================')

