# Explanation on the Model
This directory contains a pretrained model in ResNet architecture that can 
be used to do scene recognization. They are downloaded from https://github.com/CSAILVision/places365
According to the authors, "It is Preact ResNet with 50 layers. The top1 error is 44.82% 
and the top5 error is 14.71%". On the website, the authors did not show the accurarcy. However, 
as a comparison, from the same link, ResNet-152 has a performance as "on the validation set, 
the top1 error is 45.26% and the top5 error is 15.02%", yet "for 10 crop average it has 85.08% on the validation set and 85.07% on the test set for top-5". So, I assume that ResNet-50 should be good enough to carry
out jobs.
# Demonstration
There are two images taken from sequence 07, with similar scenes. You can run run_placesCNN.py to test their similarity scores. 
