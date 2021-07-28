# ---------------------------- JW, 7/28/2021 ----------------------------
Instructions for installing the necessary software/packages and setting up the virtual environment for the Object Detection GUI


## Table of Contents
* [Conda Environment Setup](#conda-environment-setup)
* [Object Detection GUI Tutorial](#object-detection-gui-tutorial)


## Conda Environment Setup

The following steps detail how to set up a Conda virtual environment. This enables python packages to be installed separately from the server-wide or network-wide packages (prevents version conflicts).
---
## 1) Ensure Conda is installed

Conda should be currently installed on the network drive. If it's not on one of the high side machines, though, there are plenty of [guides online](https://linuxize.com/post/how-to-install-anaconda-on-centos-7/) that describe how to install it.



## 2) Create a Conda environment & install Python, pip, and pyqt

To create a new conda environment, run the following commands from the command line:

```
$ conda create -n <my_env>
$ conda activate <my_env> python pip pyqt
```

Note that you may have to run "conda init <your shell>" if errors occur at this stage.


## 3) Install Packages

If this hasn't been completed yet, clone this repo. To install some of the necessary packages, navigate to the cloned repo directory and run the following from the command line:
```
$ ./install.sh
```


You should now be able to open the GUI. 



## Object Detection GUI Tutorial

The following steps outline the procedure for using the object detection GUI. This will fine-tune a pre-trained model so that it can run object detection after being trained on 10-50 images.
---


## 1) Enter your Conda environment

Make sure you've entered your Conda environment. This can be done by running
```
$ conda activate <my_env>
```


## 2) Open the GUI

Navigate to <gui_path> (the cloned repo directory) and run the following:
```
python guiObjectDetection.py
```

This should open the GUI.


## 3) Update Base Directory

TensorFlow operates on a particular directory structure called a "workspace". The base directory sets the parent directory for this structure. Feel free to enter any directory here (if no text is entered, the directory defaults to <gui_path>). The "existing workspace" indicator will glow red if no workspace currently exists in the selected base directory, and it will glow green if a workspace does exist.

*Instructions:*
```
* [Workspace Tab]
* enter base directory
* press "Update Base Directory"
```


## 4) Create Workspace (first time)

A workspace will likely only have to be generated once, but the option to generate several is still provided (some packages will be installed in this step that are workspace-dependent, so if you'd like to use newer versions of the packages, creating a new workspace is a potential option).

To generate the workspace, press the "Create Workspace" button. This creates the following directory structure (all of the following listed are directories):

    > <path_to_base_dir>
      > TensorFlow  
        > scripts
          > preprocessing  
        > workspace  

*Instructions:*
* [Workspace Tab]
* press "Create Workspace"


## 5) Collect images & draw bounding boxes

First, collect a set of images. .jpgs or pdfs are preferred. If images are contained in pdfs, select the "PDF Converter" tab in the GUI, select the desired pdf, select/create an output directory to store the images to, and set the output image name. Note that if you type "image" into the text box and there are two images found in the pdf, "image_1" and "image_2" will be saved to the output directory. Additionally, if you run the conversion with the same output image name again, the current images will not be overwritten -- the next image will be named "image_3", the one after that "image_4", and so on.

Once several images have been collected, you'll want to annotate the image with bounding boxes drawn around the objects of interest. To do this, I recommend using labelImg -- it's a Python GUI that generates .xmls with the bounding box information, and it's installed along with the workspace in the prior step.

To use labelImg, navigate to "<gui_path"/labelImg-<labelImg_ver>" and run the following:
```
$ python labelImg.py
```

From here, open an image, use the "Create\nRectBox" to draw boxes, and give the objects appropriate and consistent object class names (e.g. if you're working with images of fruits, "Orange" and "Apple" are potential object class names). Press "Change Save Dir" to store the generated .xmls in the same folder as the images. Once the boxes are all drawn, press "Save" to generate the .xml.


## 6) Generate Dataset

Datasets simply refer to a collection of images that contain a set of objects to be detected. If you have an existing dataset with 6 object classes, and you add another image with another object class, you should create a new dataset. From a TensorFlow perspective, it's an appension to the workspace.

To generate a dataset, click on the "Dataset" tab, give a name to your dataset, enter it in the "Dataset Name: " text box, and press the "Generate new Dataset" button. This creates the following directory structure:

     > <path_to_base_dir>
       > TensorFlow  
         > scripts
           > preprocessing  
         > workspace  
           > <your_dataset>
             > annotations
             > exported-models
             > images
               > test
               > train
             > models
             > pre-trained-models

*Instructions:*
* [Dataset tab]
* enter dataset name
* press "Generate New Dataset"
* select the correct dataset
* add/remove objects


## 7) Update object classes

Object classes refer to the types of objects you wish to detect. These will be based on the object class names you used to annotate the images in step [5](#collect-images-&-draw-bounding-boxes). To add/remove objects, enter the names of the object classes -- *separated by commas* -- into either the "Add Object" or "Remove Object" text box.


## 8) Add Training/Test Images

Training refers to the process of updating the model. Testing/validation refers to the process of analysis, or "validation", to ensure that the model will perform well in practice. Testing/validation is optional, but a good ratio is to use 80% of your images/xmls for training and the remaining 20% for testing.

To add images, click on the "Training Images" tab, Use the tool buttons labeled "..." to add the .jpgs and the associated .xmls to the training/testing folders in the dataset. Note that the list of images only displays the image names -- *it does not display the .xmls in the training/testing folders*. Once these are added and consistent with the list of current objects in the "Dataset" tab, press the "Generate TF Records" button.

*Instructions:*
* add training images
* add test images (optional)
* press "Generate TF Records"


## 9) Model Config

First, select a pre-trained model to fine-tune. The five currently included are "Resnet152", "EfficientDet", "Inception Resnet", "Hourglass", and "Mobilenet". Resnet and Inception tend to perform very well.

Next, create a name for the model you'll be training. This will store all of the checkpoint files/updated weights during training. Once you've entered a name, press "Create New Model". This will update the model list, and you'll be able to view the default pipeline information for the pre-trained model.

Next, update the pipeline config parameters. Below are descriptions of what each does to the neural net:
* Number of Object Classes: corresponds to the number of classes shown in the [Dataset] tab.
* Batch Size: refers to how many images are processed by the neural net before weights are updated (use between 2-32, lower if not enough RAM).
* Learning Rate: describes how significantly the weights should change during training. Larger learning rates -> larger weight updates.
* Warmup Learning Rate: this is the initial learning rate.
* Total Steps: the total number of times the weights should be updated during training. If you want to do fine-tuning, simply set this value higher than the prior number of steps used to train the model, then press "Train" again.
* Warmup Steps: start at the warmup learning rate, then slowly increase during the "Warmup Steps". Once the net has iterated through the warmup steps, uses the value set by "Learning Rate" and slowly decreases until "Total Steps" is reached.


*Instructions:*
* [Model tab]
* select a pre-trained model
* enter a model name
* press "Create New Model"
* select that model from "Your Models"
* adjust pipeline config parameters
* press "Update Config" to save values


## 10) Train Model

Make sure you have the correct model selected in the "Your Models" combo box. Once ready, press the "Train" button to train, or press "Train/Evaluate" to do training and validation simultaneously. Note that these are non-blocking processes, so *you must monitor the command line to see when training is complete* (wait for step number to reach the specified "Total Steps" in the pipeline config). If at any point you wish to stop training, press the "Stop Training" button.


*Instructions:*
* [Model tab]
* press "Train" or "Train/Evaluate" to train the model


## 11) Export the Model

This process looks at the model selected in the "Your Models" combo box, detects the most recent checkpoint, and exports those weights as a functioning model. This exported model must be given a name -- this is determined by the text in the "Name for Exported Model" text box. Once you've created a name and entered it, press the "Export*" button.


*Instructions:*
* [Model tab]
* enter name for exported model
* press "Export*"


## 12) Add New Images

Press the "Detection" tab. Now that you have an exported model, you'll likely want to see how well it performs on "new" data -- the images that you're intending to run this model on. To add images, press the "..." tool button and add the .jpgs you're interested in performing object detection on. *Note that .xmls are not required for running object detection*.


*Instructions:*
* [Detection tab]
* add "new" images


## 13) Run Object Detection

Select an exported model and set a confidence threshold (this displays bounding boxes on the image such that bbox confidence rating > threshold).
When ready, press "Run".

Once the object detection is complete, feel free to press the left and right arrows to view the images. As it currently stands, all images will have bounding boxes drawn, and then they'll be sent to ".../TensorFlow/workspace/<dataset>/images/unidentified", so if you want to move these images/view them in a better photo viewer, they're located there.


*Instructions:*
* [Detection tab]
* press "Run"
