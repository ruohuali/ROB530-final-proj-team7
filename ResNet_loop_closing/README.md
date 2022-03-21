# Explanation on the Model
This directory contains a pretrained model in ResNet architecture that can 
be used to do scene recognization. They are downloaded from https://github.com/CSAILVision/places365
According to the authors, "It is Preact ResNet with 50 layers. The top1 error is 44.82% 
and the top5 error is 14.71%". On the website, the authors did not show the accurarcy. However, 
as a comparison, from the same link, ResNet-152 has a performance as "on the validation set, 
the top1 error is 45.26% and the top5 error is 15.02%", yet "for 10 crop average it has 85.08% on the validation set and 85.07% on the test set for top-5". So, I assume that ResNet-50 should be good enough to carry
out jobs.
# Demonstration
If you want to check the effectiveness of the model, please type the command
 python run_placesCNN.py
 
 In the python file, you can edit the variable img_name to the custom image
 
 Also, I have listed four images in Sample_images, which you can try them out
