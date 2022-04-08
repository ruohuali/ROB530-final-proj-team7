import os

class Parameters():
	def __init__(self):
		self.n_processors = 2
		# Path
		self.data_dir =  '/mnt/hgfs/EECS568/kitti_dataset'
		self.image_dir = self.data_dir + '/sequences/'
		self.pose_dir = self.data_dir + '/poses/'
		
		self.train_video = ['00', '01', '02', '05', '08', '09']
		self.valid_video = ['04', '06', '07', '10']
		self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8
		

		# Data Preprocessing
		self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
		self.img_w = 608   # original size is about 1226
		self.img_h = 184   # original size is about 370
		self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
		self.img_stds =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
		self.minus_point_5 = True

		self.seq_len = (5, 7)
		self.sample_times = 3

		# Model
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0   # 0: no dropout
		self.clip = None
		self.batch_norm = True
		# Training
		self.epochs = 250
		self.batch_size = 8
		self.pin_mem = True
		self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
					# Choice:
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
		
		# Pretrain, Resume training
		self.pretrained_flownet = None
								# Choice:
								# None
								# './pretrained/flownets_bn_EPE2.459.pth.tar'  
								# './pretrained/flownets_EPE1.951.pth.tar'
		self.resume = True  # resume training
		self.resume_t_or_v = '.train'
		

par = Parameters()
