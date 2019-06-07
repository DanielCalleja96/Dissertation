# YOLOv2
Real-time object detection and classification based on [paper](https://arxiv.org/pdf/1612.08242.pdf).

The model implemented is trained and tested on 2 seperate datasets.

First model has been trained and tested on the [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset. A dataset of German traffic signs.

Second model has been trained and tested on the [LISA](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html) dataset. A dataset of USA traffic signs.

# Darkflow
This implementation is based on [Darkflow](https://github.com/thtrieu/darkflow) open source neural network framework, written in C and CUDA and supports both GPU and CPU compilation.

### Testing
To test upon this implementation you may use the file found in scripts folder by the name of YOLOv2-Prediction.py.

Set the model to which model would you like to test it upon found in the cfg folder.

Load the last checkpoint found in the ckpt folder.

Set the path on which images shall it tested against.

Important that the **labels** file is set depending on which dataset you will be testing against.

Copy the labels folder found within the files of either LISA or GTSRB, and paste them in the main folder

Once run a new file by the name of file will be created.

#### Test mAP

Once the file is created, open and save it to .txt.

Go into scripts and modify the code where necessarily according to the paths and run the Cleaning program.

Go into scripts open the GroundTruth program and modify the code where necessarily according to the path and then run the GroundTruth program.

Once both program has been run go into the mAP file, retrieved implementation from [link](https://github.com/Cartucho/mAP), and confirm that everything is in place.

You may add the images to show animation, however this is optional.

Open a CMD file in the mAP folder and run: 'python main.py'.

All results will be shown in results folder.

#### Testing Accuracy
In order to test the accuracy file, which has been retrieved from this [link](https://github.com/0merjavaid/darkflow/blob/accuracy/accuracy.py), run the file found in the scripts folder by the name of JSON-Format.

Open the JSON format program, modifiy the paths required and run the JSON Format program.

Once complete open the accuracy script change the paths were marked and run.

### Results
#### GTSRB dataset demo result



#### LISA dataset demo result


