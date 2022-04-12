# ROB530-final-proj-team19
This branch contains the codes in loop closing demo, including how we obtain the pretrained model, build the socket communication and deploy the model in the pipeline
(Xun Tu)


# Explanation on Directories 
1. demo_feature_extractor: a demo about how we use the model as a feature extractor
2. demo_loop_closing: a demo about how we deploy the model through socket programming
3. demo_resnet: a demo about how to build the model
4. expample_icp: a demo about how to build up the socket
5. demo_loop_closing algorithm: a demo about our own loop detection algorithm, though it is not deployed successfully with our DeepVO frontend. 


# Log: Modifications in Loop Closing part on ORB SLAM 2(Xun Tu)
**Update 3/27/2022
I am considering passing the name of the image that the currently evaluated frame is using to 
python model, and then generate several logits for comparison
The trajectory would be:
1. *in mono_kitti.cc, pass an extra variable as the name of the image to system "SLAM" 

2. *in System.cc/System.h, pass an extra variable as the name of the image to "mpTracking"

3. *in Tracking.cc/Tracking.h, pass an extra variable as the name of the image to "mCurrentFrame"

4. *in Frame.h/Frame.cc, add a new member as the name of the image

5. *in KeyFrame.h/KeyFrame.cc, add a new member as the name of the image

6. *in LoopClosing.cc, instead of using mBowVec, pass the name of the image of the KeyFrame to the socket and 
grab the logits

7. *in LoopClosing.cc, replace "score()" with our own way to calculate the similarity between two vectors


**Update 3/28/2022
Have completed the steps listed above in general (marked in "*"). Now, the problem is... the server would shut up immediately after it receives the data and sends it back to the client, while the client needs to send the names of the image to the server several times. We need to fix this. 

**Update 4/2/2022
Have modified the server such that it won't shut down automatically after sending one piece of data
Also modify the file KeyFrameDataBase.cc, which was omitted by me last time. Fix the function to find the candidate keyframes.

**Update 4/3/2022
Apply similar changes to Stereo examples
