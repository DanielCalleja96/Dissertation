from darkflow.net.build import TFNet
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import sys
from stopwatch import Stopwatch

options = {
    'model': 'cfg/tiny-yolo-voc-38c.cfg',
    #'load' : 'bin/tiny-yolo-voc.weights',
    'load' : 5800,    
    'threshold': 0.01,
    'gpu': 0.8
}

tfnet = TFNet(options)

mypath='./GTSRB/GTSRBTestImages/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)

stopwatch = Stopwatch() 

for n in range(0, len(onlyfiles)): 
    stopwatch.start()
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
    sys.stdout = open('file','a')
    result = tfnet.return_predict(images[n])
    if len(result) == 0:
        print("none 0 0 0 0 0")       
    else:
        print(result[0])
    #[print(a) for a in result]
    #print(result[{ 0 }])    
stopwatch.stop()
str(stopwatch)