# ROB530-final-proj-team19
A freaking cool codebase for some SLAM stuff

(Xun Tu)
Currently, the model in directory ORB-SLAM2 works well on my Ubuntu machine
(OpenCV: , Eigen: 3.10)
Sometimes I may need to remove "build" directory and rebuild everything before moving on
I would recommend to incorporate this directory directly into our project
and continue our modifications on it. 
I have not modified anything yet, except several TODOs in the files listed below


# Loop closing Modification (Xun Tu)
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

7. (optional) in LoopClosing.cc, replace "score()" with our own way to calculate the similarity between two vectors,
but in a similar manner described in OrbVocabulary.h/TemplateVocabular.h/
-->  it turns out that the method "score()" only needs two sorted and normalized vectors, which is independent of the meaning of the vectors. And it actually depends on the user's setup. So, we may just pass the two vectors from our python model to it directly

**Update 3/28/2022
Have completed the steps listed above in general (marked in "*"). Now, the problem is... the server would shut up immediately after it receives the data and sends it back to the client, while the client needs to send the names of the image to the server several times. We need to fix this. 
