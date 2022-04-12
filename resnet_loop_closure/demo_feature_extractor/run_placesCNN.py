# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from torchsummary import summary
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# Vector similarity calculation
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
# th architecture to use
arch = 'resnet50'

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


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

# load the test image
img_name1 = './000000.png'
img_name2 = './001062.png'


img1 = Image.open(img_name1)
img1 = img1.convert('RGB') #Convert it into RGB scale
input_img1 = V(centre_crop(img1).unsqueeze(0))

img2 = Image.open(img_name2)
img2 = img2.convert('RGB') #Convert it into RGB scale
input_img2 = V(centre_crop(img2).unsqueeze(0))

# forward pass
logit1 = model.forward(input_img1)
logit2 = model.forward(input_img2)
		
logit1 = F.softmax(logit1, 1).data.squeeze()
logit2 = F.softmax(logit2, 1).data.squeeze()	
lst1 = logit1.tolist()
lst2 = logit2.tolist()
print("Similarity score between two scenes is")
print(VecSim(lst1, lst2))

