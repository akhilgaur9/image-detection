# image-detection: How to build an Object Detection model using the ImageAI library
To build the object detection system,we will be using ImageAI, a python library which supports state-of-the-art machine learning algorithms for computer vision tasks.
# Creating the Anaconda environment in the system
Step 1: Create an Anaconda environment with python version 3.6.
conda create -n retinanet python=3.6 anaconda
Step 2: Activate the environment and install the necessary packages.
 activate retinanet
conda install tensorflow numpy scipy opencv pillow matplotlib h5py keras
Step 3: Then install the ImageAI library.
pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl
Step 4: Now download the pretrained model required to generate predictions. This model is based on RetinaNet Pretrained model 
Step 5: Copy the downloaded file to the current working folder
Step 6: Download the image from the link.https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/06/I1_2009_09_08_drive_0012_001351-768x223.png 
Name the image as image.png.
Step 7: Open jupyter notebook (type jupyter notebook in the terminal) and run the following codes:
from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=False) # detecting person from the image.
detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "image.png"), output_image_path=os.path.join(execution_path , "image_new.png"), custom_objects=custom_objects, minimum_percentage_probability=65)#setting the threshold probability at 65
for eachObject in detections:
print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
print("--------------------------------")
#create a modified image file named image_new.png, which contains the bounding box for the image.
Step 8: To print the image use the following code:
from IPython.display import Image
Image("image_new.png")
