import socket
import struct
import numpy as np
import sys
from deepvo.helper import eulerAnglesToRotationMatrix, R_to_angle

def toSE3(vo_pose):
    pose = np.zeros((4,4))
    pose[3,3] = 1
    pose[:3,:3] = vo_pose
    return pose

def floatList2Bytes(lst):
    buf = bytes()
    for val in lst:
        buf += struct.pack('d', val)
    return buf

np.set_printoptions(suppress=True) # remove scientific notation when printing
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8080  # Port to listen on (non-privileged ports are > 1023)
seq_num = sys.argv[1]

rel_poses = np.genfromtxt('deepvo/poses/'+seq_num+'_rel.txt')
abs_poses = np.genfromtxt('deepvo/poses/'+seq_num+'_abs.txt')
N = rel_poses.shape[0]
rel_poses = rel_poses.reshape(N, 3, 4)
abs_poses = abs_poses.reshape(N, 3, 4)

# Set up socket to client
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))

while True:

    # Recieve message from ORB SLAM
    s.listen()
    conn, addr = s.accept()
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
    print('Received idx:', img_idx)
    print('Previous pose recieved: ', prev_pose)
    if not data:
        break
    
    predict_pose_seq = R_to_angle(rel_poses[img_idx])[:6]
    # Transform x,y,z using yaw of vehicle
    ang = eulerAnglesToRotationMatrix([0, prev_pose[0], 0])
    location = ang.dot(predict_pose_seq[3:])
    predict_pose_seq[3:] = location

    # Add relative pose to absolute
    new_pose = [a + b for a, b in zip(predict_pose_seq, prev_pose)]
    new_pose[0] = (send_mTwc[0]+np.pi)%(2*np.pi)-np.pi # normalize to [-pi,pi] over y-axis
    send_mTwc = np.zeros(3,4)
    send_mTwc[:3,:3] = eulerAnglesToRotationMatrix(new_pose[:3])
    send_mTwc[:3,3] = new_pose[3:]
    send_mTwc = toSE3(send_mTwc)
    send_mTcw = np.linalg.inv(send_mTwc)
    print('Absolute pose at ', img_idx)
    print(send_mTcw)

    # Send pose to ORB SLAM
    abs_pose_lst_bytes = floatList2Bytes(send_mTcw.reshape(-1).tolist())
    conn.sendall(abs_pose_lst_bytes)

    print('=================================================')

