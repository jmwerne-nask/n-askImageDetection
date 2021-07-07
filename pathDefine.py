



# edit these as necessary
TFDir = '~/TensorFlow'
datasetDir = TFDir + '/workspace/SVS'
modelDir = 'models/resnet_Colab/v1'
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
