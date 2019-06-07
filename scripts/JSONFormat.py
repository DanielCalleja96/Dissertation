from darkflow.net.build import TFNet
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import sys

options = {
    'model': 'cfg/tiny-yolo-voc-43c.cfg',
    #'load' : 'bin/tiny-yolo-voc.weights',
    'load' : 6615,    
    'threshold': 0.01,
    'gpu': 0.8
}

tfnet = TFNet(options)

mypath='C:/dark/darkflow-master/LISA/LisaTestImages'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)

for n in range(0, len(onlyfiles)):   
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
    sys.stdout = open('Tiny-YOLO-DetectionFile-JSON','a')
    result = tfnet.return_predict(images[n])
    if len(result) == 0:
        print("{\"label\": \"none\", \"confidence\": 0, \"topleft\": {\"x\": 0, \"y\": 0}, \"bottomright\": {\"x\": 0, \"y\": 0}}")       
    else:
        print(result[0])
    #[print(a) for a in result]
    #print(result[{ 0 }])    
    