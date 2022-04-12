import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from torchsummary import summary
import time
import numpy as np
import matplotlib.pyplot as plt
# th architecture to use
arch = 'resnet50'
   

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
def VecSim(lst1, lst2):
	"""
	Compare the similarity between two lists using L1 norm
	"""
	s = len(lst1)
	score = 0
	for i in range(s):
		v = lst1[i]
		w = lst2[i]
		score = score + abs(v - w) - abs(v) - abs(w)

	score = -score/2.0
	return score


def ExtractFeatures(seq, dataset_path):
	# load the pre-trained weights
	model_file = '%s_places365.pth.tar' % arch
	if not os.access(model_file, os.W_OK):
	    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
	    os.system('wget ' + weight_url)

	model = models.__dict__[arch](num_classes=365)
	checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)

	model.fc = Identity()
	model.eval()

	# load the class label
	file_name = 'categories_places365.txt'
	if not os.access(file_name, os.W_OK):
	    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
	    os.system('wget ' + synset_url)
	classes = list()
	with open(file_name) as class_file:
	    for line in class_file:
	        classes.append(line.strip().split(' ')[0][3:])
	classes = tuple(classes)

	# load the image transformer
	centre_crop = trn.Compose([
	        trn.Resize((256,256)),
	        trn.CenterCrop(224),
	        trn.ToTensor(),
	        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])


	# Find the directory of the sequence
	if seq < 10:
		seq_directory = dataset_path + "0" + str(seq) + "/image_0/"
	else:
		seq_directory = dataset_path + str(seq) + "/image_0/"

	# frame logits as well as the keyframes
	frames_logits = []
	keyframe_idx = []
	keyframe_insert = True
	keyframe_curr = 0

	frame_idx = 0
	frame_acc = 0

	print("Start processing all images")
	print("May take a while (around 1~6 mins, depending on the sequence) ...")
	start_time = time.time()
	for filename in os.listdir(seq_directory):
		# load the test image
		img_name = os.path.join(seq_directory, filename)
		if not os.access(img_name, os.W_OK):
			frames_logits = [-1]
			keyframe_idx = [-1]
			return frames_logits, keyframe_idx


		img = Image.open(img_name)
		img = img.convert('RGB') #Convert it into RGB scale
		input_img = V(centre_crop(img).unsqueeze(0))
				
		# forward pass
		logit = model.forward(input_img)
		#For some types of scoring method, we need sorted and normalized vector; 
		#logit, _ = logit.sort(stable=True)
		#n = torch.linalg.vector_norm(logit, ord=1, dim = 1)
		#logit = logit/n
		#logit = logit.squeeze()
		logit = F.softmax(logit, 1).data.squeeze()
				
		lst = logit.tolist()
		frames_logits.append(lst)
		score = VecSim(frames_logits[keyframe_curr], lst)
		if keyframe_insert: # If allowed to insert a keyframe, record it
			keyframe_idx.append(frame_idx)
			keyframe_curr = frame_idx
			keyframe_insert = False
			frame_idx = frame_idx + 1
			frame_acc = 0
		elif frame_acc >= 20 and VecSim(frames_logits[keyframe_curr], lst) < 0.9:
			keyframe_insert = True 
			# If there are more than 20 frames and the newest frame doesn't "look" like the current keyframe,
			# insert a new one
			frame_idx = frame_idx + 1
			frame_acc = frame_acc + 1
		else:
			# Else, you do nothing and continue counting the frames
			frame_idx = frame_idx + 1
			frame_acc = frame_acc + 1

	print("All images processed in the Sequence!")
	print("--- %s mins ---" % ((time.time() - start_time)/60))
	return frames_logits, keyframe_idx

def LoopDetection(frames_logits, keyframe_idx, pose_path):
	l = len(keyframe_idx)
	loop_detected = []
	initial_poses = np.genfromtxt(pose_path)
	for i in range(l):
		pose_i = [initial_poses[keyframe_idx[i], 3], initial_poses[keyframe_idx[i], 11]]
		start = keyframe_idx[i]

		# Evaluate the similarity scores within the group; find the lowest
		minScore = 1
		start = keyframe_idx[i]
		if i == l-1:
			end = l
		else:
			end = keyframe_idx[i+1]
			for g in range(start, end):
				score = VecSim(frames_logits[g], frames_logits[start])
				if score < minScore:
					minScore = score
		# Evaluate the previous keyframes
		best_loop_score = minScore
		best_loop_cand = -1
		for k in range(i): 
			pose_k = [initial_poses[keyframe_idx[k], 3], initial_poses[keyframe_idx[k], 11]]
			dist = (pose_i[0] - pose_k[0]) ** 2 + (pose_i[1] - pose_k[1]) ** 2
			if dist < 20 and abs(k-i) > 5: #If the two places are close enough
				score_keyframes = VecSim(frames_logits[keyframe_idx[k]], frames_logits[start])
				#if (abs(k-i) > 5):
					#print("distance:")
					#print(dist)
				# For the potential loop candiates, evaluate the similarities upon 
				# the current keyframe and this keyframe
				# Find the one with the highest similarity score
				if score_keyframes > best_loop_score:
					#print("Similarity score")
					#print(score_keyframes)
					#print("KeyFrames: ")
					#print(str(keyframe_idx[k]) + " " + str(keyframe_idx[i]))
					best_loop_score = score_keyframes
					best_loop_cand = k
		# For the found keyframe, search through its associative members to find the most appropriate one
		if best_loop_cand > -1: 
			best_loop_member = best_loop_cand
			best_loop_member_score = best_loop_score
			for m in range(keyframe_idx[best_loop_cand], keyframe_idx[best_loop_cand+1]):
				score_member = VecSim(frames_logits[m], frames_logits[start])
				if score_member > best_loop_member_score:
					best_loop_member = m

			loop_detected.append((start, best_loop_member))
		

		

	isLoop = not(len(loop_detected) == 0)
	return isLoop, loop_detected

seq_number = 10 # sequence number
if seq_number < 10:
	seq_str = "0" + str(seq_number)
else:
	seq_str = str(seq_number)
dataset_path = "/home/xun/data_odometry_gray/dataset/sequences/" #Your path to dataset
pose_path = "./initial_poses/" + seq_str + ".txt" #Your path to the initial guess of poses; 
# To the demo purpose, I am just using GT poses here
# Just want to show that the algorithm is working qualitatively

frames_logits, keyframe_idx = ExtractFeatures(seq_number, dataset_path)
start = time.time()
print("Evaluating the potential loops...")
isLoop, loop_detected = LoopDetection(frames_logits, keyframe_idx, pose_path)
loop_detected = np.array(loop_detected)
trajectory = np.genfromtxt(pose_path)


xi = trajectory[:,3]
yi = trajectory[:,11]
if isLoop:
	x1 = trajectory[loop_detected[:,1], 3]
	y1 = trajectory[loop_detected[:,1], 11]
	x2 = trajectory[loop_detected[:,0], 3]
	y2 = trajectory[loop_detected[:,0], 11]

	plt.plot(xi, yi, label="Ground truth Trajectory")
	plt.plot(x1, y1, 'bo', label="Found loop candidates (former frame)")
	plt.plot(x2, y2, 'ro', label = "Found loop candidates (latter frame)")
	plt.legend(loc="upper left")
	plt.show()
else:
	plt.plot(xi, yi, label="Ground truth Trajectory")
	plt.show()
print("Loop closing complete!")
print("Keyframes")
print(keyframe_idx)
print("Has a loop?")
print(isLoop)
print("End of loop closing")
