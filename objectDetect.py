import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import sys
import re
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt

from pathDefine import exportModelDir
from pathDefine import savedModelPath
from pathDefine import newImageDir as img
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('imageDir', None, 'Specified image directory')
flags.DEFINE_string('imagePath', None, 'Specified image path')
FLAGS(sys.argv)
#print(FLAGS.imagePath)
if FLAGS.imagePath:
    img = [FLAGS.imagePath]

if FLAGS.imageDir:
    img = []
    for filename in sorted(os.listdir(FLAGS.imageDir)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            fullfilename = os.path.join(FLAGS.imageDir, filename)
            img.append(fullfilename)




#loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("/ohio/home/jmwerne/TensorFlow/workspace/SVS/annotations/label_map.pbtxt",use_display_name=True)


print(img)
count = 0

print('Loading model...', end='')
# Load saved model and build the detection function
detect_fn=tf.saved_model.load(savedModelPath)
print('Done!')

from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))
i = 101;
outputdir = '~/Tensorflow/workspace/SVS/images/svs_' + str(i) + '.png'
for image_path in img:
    print('Running inference for {}... '.format(image_path), end='')
    image_np=load_image_into_numpy_array(image_path)
    input_tensor=tf.convert_to_tensor(image_np)
    print(input_tensor.shape)
    input_tensor=input_tensor[tf.newaxis, ...]
    detections=detect_fn(input_tensor)
    num_detections=int(detections.pop('num_detections'))
    detections={key:value[0,:num_detections].numpy()
                   for key,value in detections.items()}
    detections['num_detections']=num_detections
    detections['detection_classes']=detections['detection_classes'].astype(np.int64)
    image_np_with_detections=image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
         image_np_with_detections,
         detections['detection_boxes'],
         detections['detection_classes'],
         detections['detection_scores'],
         category_index,
         track_ids=np.array([98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]),
         use_normalized_coordinates=True,
         line_thickness=5,
         max_boxes_to_draw=300,     
         min_score_thresh=.75,      
         agnostic_mode=False)
    if np.allclose(image_np_with_detections, image_np):
        print('New object detected!')
        outputdir = '/ohio/home/jmwerne/TensorFlow/workspace/SVS/images/Output/svs_'+str(i)+'.png'
        viz_utils.save_image_array_as_png(
            image_np_with_detections,
            outputdir)
        i = i+1

    #%matplotlib inline
    plt.figure()
    plt.imshow(image_np_with_detections)
    
    print('Done')
    plt.show()
