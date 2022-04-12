# Overview of the directory
This directory explains how we integrate the pretrained model into the ORB-SLAM2 pipeline using socket programming
# Explanation on the Model
This directory contains a pretrained model in ResNet architecture that can 
be used to do scene recognization. They are downloaded from https://github.com/CSAILVision/places365
According to the authors, "It is Preact ResNet with 50 layers. The top1 error is 44.82% 
and the top5 error is 14.71%". On the website, the authors did not show the accurarcy. However, 
as a comparison, from the same link, ResNet-152 has a performance as "on the validation set, 
the top1 error is 45.26% and the top5 error is 15.02%", yet "for 10 crop average it has 85.08% on the validation set and 85.07% on the test set for top-5". So, I assume that ResNet-50 should be good enough to carry
out jobs.
# Explanation on the Socket Communication
In this folder, a simple demonstration on my idea to integrate the model into orb-slam pipeline
The client will send the name of the image to python file containing the model,
and then the python file would send the logits back.
About how to use them, please check the folder "example_icp" on how to set up the client
and the server. 
**Attention** When running the executable "./client", please attach the name of the image
at argv[1]

# Explanation on the Files
Please notice that our server should hold on after it sends back the data. So, I have attached two versions of server here:
server.py -> the server that would shut down immediately after it sends back the data. Just for demonstration. 
server_used.py -> the server that would hold on. This is the one used in our module


