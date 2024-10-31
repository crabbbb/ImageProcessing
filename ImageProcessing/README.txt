To install:
pip install tensorflow==2.15.0

pip install h5py==3.9.0

pip install streamlit

--------------------------------------------
There are 3 training file:
1. Color_Segmentation_Training.ipynb
2. Edge_Canny_Training.ipynb
3. Texture_Analysis_Training.ipynb

And there is no need to train them. Just see it as a reference on how we are processing the image as the model has trained and can be used anytime.

The 3 models are:
1. color_segmentation_apple.h5
2. edge_canny_apple.h5
3. texture_apple.h5

------------------------------------------------------------------------------
In runStreamLit.ipynb, 

1. The first cell has declared the model needed to use, and it is better to use whole path than relative path.

2. In the 6th cell, inside def main(), there will be 3 Cascade Classifier declared there and it is better to use whole path than relative path. 

----------------------------------
Sometimes it cannot execute runStreamLit.ipynb, the Solution:

1. If you could not execute the runStreamLit.ipynb, then kindly use runStreamLit2.py to execute with command of "streamlit run ___whole file path_____" in terminal

--------------------------------
After executing the runStreamLit.ipynb or runStreamLit2.py, there will be a GUI pop out. In the input field, please kindly type in the full path of the image like 
"C:\\Users\\LAPTOP\\Desktop\\ImageProcessing\\testData\\2.jpg"
and then choose the processing algorithm you want.