# Master Thesis - Bag of Features for Text Detection in Natural Scene Images
![Master Thesis Demo](presentations/pics/masterProjectDemo.gif)

The primary goal of the thesis was to perform a comparative study of text detection in natural scene images using learned and hand-crafted feature descriptor. Scale Invaraint Feature Transform (SIFT) descriptor was used as the hand-crafted feature descriptor. The Maximally Stable Extremal Regions (MSER) were used to detect the scales of the text present in the image automatically.  A detailed study was performed on the effect of automatic scale detection on both text detection using learned features and hand-crafted features.

**Folders:**
- **presentations** -  contains presentation prepared during my Master thesis. 
Includes a small video (./pics/URHere.avi) displaying the text detection predictions 
by the generated model.
- **report** - Final Master thesis report
- **repository** - Code repository (includes python and matlab code)
The coding is mostly carried out in python using Scipy, Numpy, OpenCv and Vlfeat
The code contains Spherical K-means implementation and also GAP statistic implementation.
Spherical K-means is used to generate visual vocabulaory in Bag of Feature paradigm. 
GAP statistic is used to determine the optimum number of clusters.
