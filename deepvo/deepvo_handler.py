import torch
import numpy as np
from deepvo.params import par
from deepvo.model import DeepVO
from deepvo.data_helper import get_data_info, ImageSequenceDataset
from deepvo.helper import eulerAnglesToRotationMatrix

class DeepVOHandler():
    def __init__(self, seq_num, seq_len=6, overlap=5, batch_size=1):
        self.M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
        self.M_deepvo.load_state_dict(torch.load('deepvo/model/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train', map_location={'cuda:0': 'cpu'}))
        self.M_deepvo.eval()
        df = get_data_info(folder_list=[seq_num], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        self.dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

    # img_idx: index of KITTI image to be processed
    # prev_pose: [theta_y, theta_x, theta_z, x, y, z]
    def get_pose(self, img_idx, prev_pose):
        # Index differently in first sequence
        if img_idx < 5:
            ds_idx = 0
            pose_idx = img_idx
        else:
            ds_idx = img_idx-5
            pose_idx = -1
        
        # Pass through DeepVO to get relative pose for all seq in seq_len
        x = self.dataset[ds_idx][1]
        x = x[None,:] # mimic batch size of 1
        batch_predict_pose = self.M_deepvo.forward(x).data.cpu().numpy()
        
        # Only keep relevant idx in sequence
        predict_pose_seq = batch_predict_pose[0][pose_idx]
        rel_pose = predict_pose_seq

        # Transform x,y,z using yaw of vehicle
        ang = eulerAnglesToRotationMatrix([0, prev_pose[0], 0]) # prev pose as R
        location = ang.dot(predict_pose_seq[3:]) # R * rel_pose_xyz
        predict_pose_seq[3:] = location

        # Add relative pose to absolute
        cur_pose = [a + b for a, b in zip(predict_pose_seq, prev_pose)]
        cur_pose[0] = (cur_pose[0]+np.pi)%(2*np.pi)-np.pi # normalize to [-pi,pi] over y-axis

        return rel_pose, cur_pose
    
    def get_len(self):
        return len(self.dataset)
