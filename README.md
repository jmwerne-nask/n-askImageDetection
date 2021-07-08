# ---------------------------- JW, 7/07/2021 ----------------------------
# Instructions for training/fine-tuning a pre-trained neural network to recognize objects in a dataset


---
## The following should be run any time you're installing TensorFlow and/or the Object Detection API on a new machine


## 1) Prepare directory structure

Most of the scripts written for the Object Detection API require a particular file structure in order to operate correctly. As a result, find a suitable directory to store the bulk of this project (the home area works well). For the remainder of this README, I will denote this directory as ```<PATH_TO_TF>```.

Create the following folder structure within ```<PATH_TO_TF>```:

    > TensorFlow  
      > scripts
        > preprocessing  
      > workspace  
        > <dataset> (name this whatever you want)  
          > annotations  
      	  > exported-models
          > images
            > test
	    > train
          > models
          > pre-trained-models


You'll want to download the script [generate_tfrecord.py](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py) [1], and you'll want to save it in the folder ```<PATH_TO_TF>/TensorFlow/scripts/preprocessing```.

[1] https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py



## 2) Enter Anaconda Virtual Environment

I chose to use the Anaconda virtual environment as an interface to the Python libraries required for this task. Activate the Anaconda virtual environment by running

    $ conda activate <my_env>

where ```<my_env>``` is your already-created environment (if you have questions about installing conda/creating an environment, consult google or the "labelImgInstallInstructions.txt" document)


If you haven't already installed pip, run

    $ conda install pip

## 3) Install TensorFlow & the Object Detection API

Run the following in the terminal:
    
    $ pip install --ignore-installed --upgrade tensorflow==2.5.0

Now run 
    
    $ python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

It'll output quite a few warnings, but try to look for the following output: 

    >> "tf.Tensor(1620.5817, shape=(), dtype=float32)"

If you see this text, TensorFlow was installed correctly.


Next, navigate to the "TensorFlow" folder. Download the [TensorFlow Model Garden](https://github.com/tensorflow/models) by running the following:
    
    $ git clone https://github.com/tensorflow/models



## 4) Install/Compile Protobuf & COCO API

Navigate to the [protoc releases](https://github.com/protocolbuffers/protobuf/releases) page [2]. Download the latest ```protoc-\*-linux-\*.zip``` folder. Extract the contents to a folder of your choice -- for the sake of this example, I extracted it to "<PATH_TO_TF>/TensorFlow/protoc-3.17.3". 

Once this is done, cd into "TensorFlow/models/research" and run the following to compile protobuf:
    
    $ cd <PATH_TO_TF>/TensorFlow/models/research
    $ <PATH_TO_TF>/TensorFlow/protoc-3.17.3/bin/protoc object_detection/protos/*.proto --python_out=.

To install COCO API, navigate to a directory of your choice and run the following commands. For the sake of this example, I navigated to "<PATH_TO_TF>/TensorFlow/" and ran the following:
    
    $ pip install cython
    $ git clone https://github.com/cocodataset/cocoapi.git
    $ cd cocoapi/PythonAPI
    $ make
    $ cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research

To install the Object Detection API, run the following:
    
    $ cd <PATH_TO_TF>/TensorFlow/models/research
    $ cp object_detection/packages/tf2/setup.py .
    $ python -m pip install --use-feature=2020-resolver .

To test the installation, run the following from within "Tensorflow/models/research":
    
    $ cd <PATH_TO_TF>/TensorFlow/models/research
    $ python object_detection/builders/model_builder_tf2_test.py

It should run 24 tests and end with "OK".

[2] https://github.com/protocolbuffers/protobuf/releases



---

## The following steps should be run any time you want to start with a new dataset or update an existing dataset


## 5) Copy Scripts

Navigate to the ```.../workspace/dataset``` directory (if this doesn't exist, create it, with "dataset" being replaced with a name of your choice that describes the images you're feeding the neural net). Remember: you might've called this folder something other than "dataset" -- it's the folder that contains annotations, images, models, etc.
Run the following commands to copy the train and export scripts to your main area:

    $ cp <PATH_TO_TF>/Tensorflow/models/research/object_detection/model_main_tf2.py .
    $ cp <PATH_TO_TF>/Tensorflow/models/research/object_detection/exporter_main_v2.py .

Additionally, you'll want to download the following scripts for setting path variables and performing object detection on images:

    $ git clone https://github.com/jmwerne-nask/n-askImageDetection/blob/main/pathDefine.py
    $ git clone https://github.com/jmwerne-nask/n-askImageDetection/blob/main/objectDetect.py
    


## 6) Prepare dataset & Create Label Map:

First, gather images that contain the objects you're interested in classifying/detecting. Try to ensure that these images are indicative of the images you'll actually be collecting in practice (i.e. if you expect to use this neural network on high-resolution/quality images, you should gather high-resolution/quality images).

Next, open labelImg and begin drawing bounding boxes around the objects of interest. Annotate these objects accordingly. Save them to ```workspace/images/train```. Set aside about 20% of your annotated images, and place these in ```workspace/images/test```.

You'll also need to create a label map. To do this, create a new file called "label_map.pbtxt" and structure it as follows:

    item {
          id: 1
          name: 'object1'
    }
    item {
          id: 2
          name: 'object2'
    }

... and so on.



## 7) Generate Image Records

Navigate to ```.../workspace/dataset``` and run the following commands to generate the train and test records:
    
    $ python <PATH_TO_TF>/TensorFlow/scripts/preprocessing/generate_tfrecord.py -x ./images/train -l 
      ./annotations/label_map.pbtxt -o ./annotations/train.record      
    $ python <PATH_TO_TF>/TensorFlow/scripts/preprocessing/generate_tfrecord.py -x ./images/test -l 
      ./annotations/label_map.pbtxt -o ./annotations/test.record

Make sure that both train.record and test.record are nonempty, otherwise the training won't work (and there's a chance that it might not throw an error).


---


## The following steps should be run any time you want to train a new model


## 8) Download Pre-Trained Model

Next, download a pre-trained model from the [object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Extract it to the ```pre-trained-models``` folder. I'd recommend using Faster RCNN Resnet, as it trains relatively quickly and has really good training accuracy.

From here, **create a folder within "models"**. Name it something like "my_resnet" or "model_1" -- for the sake of this readme, I will denote this folder as ```<your_model_name>```. **Take the pipeline.config file** within your recently extracted "pre-trained-models" folder and **copy it to this folder**. 

[3] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md



## 9) Edit pipeline.config

There are several parameters that need to be set in this file before any training can begin. Make the following necessary adjustments (note that the config files differ from model to model, so the line numbers are approximates):

    > Line    3: set the number of object classes
    > Line ~130: set batch size (if low on RAM, set to 1 or 2; if >12GB of RAM, depending on the pre-trained model, can 
                    use 8 or larger for batch size)
    > Line ~160: set fine_tune_checkpoint to the following file:
                    "pre-trained-models/<model_name_here>/checkpoint/ckpt-0".
    > Line ~160: set num_steps to the maximum number of steps you wish to use to train your model. I've been using 
                    between 5000-to-15000 for my dataset of 20 images, but this is a parameter you'll likely want to 
		        tweak. Note: if you want to stop training, then continue training the model later on, you can do 
		        so by simply changing this parameter and rerunning the script in step 5).
    > Line ~170: set fine_tune_checkpoint_type: "detection"
    > Line ~170: set use_bfloat16: false (only set to true if using TPU)
    > Line ~170: set label_map_path: "annotations/label_map.pbtxt"
    > Line ~170: set input_path: "annotations/train.record"
    > Line ~180: set label_map_path: "annotations/label_map.pbtxt"
    > Line ~180: set input_path: "annotations/test.record"


Feel free to tweak any of the other parameters as well, but remember that most are initialized to help the model perform well immediately. I find that tweaking the learning rate has the biggest impact on both convergence and performance of the model.



## 10) Set Path/Directory Variables

Open the file named ```.../workspace/<dataset>/pathDefine.py``` and edit the path names. You'll most likely need to alter the following variables:

  - modelDir -- this determines where the checkpoints get saved to during training, as well as the pipeline that the training script uses.
  - exportModelDir -- this determines where the exported model gets saved to.
  - newImageDir -- (optional) this determines which images you want the exported model to perform object detection upon. These are the "new" images that you'd actually collect in the field for detection/classification.



## 11) Train the Model 

**Note**: if you want to evaluate your model, read through 11 ii) before executing any commands in this section.

Navigate to ```.../workspace/<dataset>```. Run the following command:

    $ python model_main_tf2.py --model_dir="models/<your_model_name>" 
      --pipeline_config_path="models/<your_model_name>/pipeline.config"      

Note: ```<your_model_name>``` was set in step 8.

You've begun training your model! If you'd like to start/stop training at any point, keep in mind that the Object Detection API stores checkpoints, so all you have to do is update the desired number of steps in the pipeline.config file, and the script automatically detects your latest checkpoint and continues training from there. Generally, you want to train until the total loss is consistently below 0.3.


## 11 ii) Evaluate the Model (optional)

In addition to training your model, you can run evaluation on the "test" dataset you created earlier -- this gives you an idea of how well your model will perform on "new" data.

To evaluate your model, open a new terminal **before/while your model is being trained**. Make sure you're in the ```.../workspace/<dataset>``` folder. Run the following:

    $ python model_main_tf2.py --model_dir="models/<your_model_name>" 
      --pipeline_config_path="models/<your_model_name>/pipeline.config"      
      --checkpoint_dir="models/<your_model_name>"
      

## 11 iii) View Model Progress (optional)

To view your model progress, open another terminal and navigate to ```models/<your_model_name>```. From here, run the following:

    $ tensorboard --logdir .
    
Once this is done, open a web browser and type ```localhost:6006``` into the url search bar. Your plots should display.


---

## The following steps should be run any time you want to export and/or use a model


## 12) Export the Model

 Once you're finished training the model, make sure you're in the ```.../workspace/<dataset>``` directory, then run the following command to export it:
 
    $ python exporter_main_v2.py --input_type image_tensor --pipeline_config_path "models/<your_model_name>/pipeline.config"      
      --trained_checkpoint_dir "models/<your_model_name>" --output_directory "exported-models/<your_name_for_the_model>"

Note: feel free to set the output_directory flag to whatever folder you want. For the sake of consistency, though, I recommend the above.



## 13) Load Images & Run the Model

If you want to test your model on some images, navigate to ```.../workspace/<dataset>``` and run the following script:

    $ python objectDetect.py

This will run object detection on the images specified in either the "pathDefine.py" script or as command line arguments. Note that you must specify an output directory in the pathDefine.py file -- this is needed because this objectDetect.py script needs a location to save the "no objects detected" images.

**Alternative uses:**
 If you want to run the model on one image, run:
 
    $ python objectDetect.py imagePath <PATH_TO_IMAGE>

 If you want to run the model on a folder full of images (must be .jpgs, .jpegs, or .pngs), run:
 
    $ python objectDetect.py imageDir <IMAGE_DIRECTORY>




**Important links:**

    [1] https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py
    [2] https://github.com/protocolbuffers/protobuf/releases
    [3] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
