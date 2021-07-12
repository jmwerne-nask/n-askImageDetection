

import sys
import os

# edit these as necessary
pathtoTF = sys.argv[1]
dataset = sys.argv[2]
TFDir = pathtoTF + '/TensorFlow'
datasetDir = TFDir + '/workspace/' + 'dataset'

model = sys.argv[3]
ver = sys.argv[4]
modelDir = 'models/' + sys.argv[3] + '/' + sys.argv[4]
configPath = modelDir + '/pipeline.config'
exportModelDir = 'exported-models/resnet_Colab/v1'

newImageDir = ['/ohio/home/jmwerne/TensorFlow/workspace/SVS/images/test/svs_55.jpg',
               '/ohio/home/jmwerne/TensorFlow/workspace/SVS/images/test/svs_56.jpg']


# these don't need to be edited unless you choose to considerably deviate from the recommended folder structure
labelPath = datasetDir + '/annotations/label_map.pbtxt'
trainImageDir = datasetDir + '/images/train'
testImageDir = datasetDir + '/images/test'
trainRecordPath = datasetDir + '/annotations/train.record'
testRecordPath = datasetDir + '/annotations/test.record'
savedModelPath = exportModelDir + '/saved_model'
