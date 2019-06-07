import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
#import socket

#setting up the model and training set
options = {
    'model': 'cfg/yolov2-voc-38c.cfg',
    'load': 2600,
    'threshold': 0.1,
    'gpu': 0.8
}

url='http://192.168.1.108:8080/video'
port = 5555

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(url)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

print("Success binding")    

while True:
    stime = time.time()  
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)    
            
        for color, result in zip(colors, results):                    
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])     
            label = result['label']
                
            confidence = result['confidence']
                        
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            height, width, channels = frame.shape
            
            frame = cv2.putText(
                    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, 
                    (0, 0, 0), 2)
                      
            cv2.imshow('frame', frame)
            
    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()