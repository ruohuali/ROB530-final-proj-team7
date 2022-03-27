import socket
import struct
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from torchsummary import summary
# th architecture to use
arch = 'resnet50'
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8080  # Port to listen on (non-privileged ports are > 1023)

def pseudoFloatList(length):
    return [x / 0.3 for x in range(length)]

def floatList2Bytes(lst):
    buf = bytes()
    for val in lst:
        buf += struct.pack('d', val)
    return buf    



# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
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

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind((HOST, PORT))
	s.listen()
	conn, addr = s.accept()
	with conn:
		print(f"Connected by {addr}")
		data = conn.recv(200)
		print(data)

		print("Image received")
		# load the test image
		img_name = data.decode('UTF-8')
		print("Handled image:")
		print(img_name)
		if not os.access(img_name, os.W_OK):
			img_url = 'http://places.csail.mit.edu/demo/' + img_name
			os.system('wget ' + img_url)
		print("Processing image:")
		img = Image.open(img_name)
		input_img = V(centre_crop(img).unsqueeze(0))

		# forward pass
		logit = model.forward(input_img)
		h_x = F.softmax(logit, 1).data.squeeze()
		probs, idx = h_x.sort(0, True)

		print('{} prediction on {}'.format(arch,img_name))
		# output the prediction
		for i in range(0, 5):
			print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
		lst = probs[0:5].tolist()
		msg = floatList2Bytes(lst)
		conn.sendall(msg)
		print("all float sent")
