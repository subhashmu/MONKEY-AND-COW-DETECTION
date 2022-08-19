# MONKEY DETECTION

## MONKEY DETECTION ON RAILWAY TRACKS MODEL

## AIM AND OBJECTIVES

## Aim
To create MONKEY Detection on Railway Tracks model which will detect
MONKEY crossing on Railway Tracks and then convey the message on the
viewfinder of camera in real time when MONKEY and COW are crossing the Tracks.

## Objectives

• The main objective of the project is to create a program which can be either run on
Jetson nano or any pc with YOLOv5 installed and start detecting using the camera
module on the device.

• Using appropriate data sets for recognizing and interpreting data using machine
learning.

• To show on the optical viewfinder of the camera module when MONKEY 
cross Railway Tracks.

## ABSTRACT

• MONKEY are detected when they are crossing the railway tracks and then
shown on the viewfinder of the camera.

• Our Team in SparrowAi have completed this project on jetson nano which is a very small
computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine
Learning (ML), where machines are trained to identify various objects from one another.
Machine Learning provides various techniques through which various objects can be
detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small
size trained model and makes ML integration easier.

• Over 32000 animals killed on railway tracks between 2016 and 2018.

• While in 2016, 7,945 animals were mowed down by trains, in 2017, the number rose to
11,683 and in 2018, it was 12,625 bringing the total number of animals killed between
2016 and 2018 to 32,253.

## INTRODUCTION

• This project is based on MONKEY detection model. We are going to
implement this project with Machine Learning and this project can be even run on jetson
nano which our team at SparrowAi has already done.

• This project can also be used to gather information about how many animals are
crossing the Railway Tracks in a given time.

• MONKEY can be classified into whether they are standing close to railway
tracks or are crossing them based on the image annotation we give in roboflow.

• MONKEY detection becomes difficult sometimes on account of various
conditions like rain, fog, night time making MONKEY Detection harder for
model to detect. However, training in Roboflow has allowed us to crop images and
change the contrast of certain images to match the time of day, lighting for better
recognition by the model.

• Neural networks and machine learning have been used for these tasks and have
obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for MONKEY detection on Railway
Tracks as well.

## LITERATURE REVIEW

• A senior railway official acknowledged the problem and said that while the number of
train accidents was decreasing, the number of animal deaths on tracks which were
around 3,000-4,000 in 2014-2015 has been increasing, which is a cause of concern.

• Trains are also bearing the brunt of cattle run over cases. Earlier this year, newly
launched Vande Bharat Express, was hit by stray cattle and its aerodynamic nose, which
is made of steel with a fibre cover on it, had to be replaced.

• West Central Railways recorded the highest number of animal deaths. South Central
Railway which covers most portions of the two Telugu states witnessed 110 deaths so
far in 2019.

• Wildlife experts say that in 10 years, the deaths of animals have soared. The reason
behind such incidents is the countrywide increase in high speed rail infrastructure. Many
of these projects cut through sensitive natural zones where animals live.

• Experts said the West Central Railway, which accounted for the highest number of
animal deaths, passeed through one of the biggest forest ranges which included the
Satpura National Park, Vindhyachal, Bharatpur, Satna and Jabalpur. South Western
Railway recorded the least number of death because it passed through the ghats which
had many bridges. There were many water bodies in Western Ghats so that animals
don’t have to cross the railway lines.

• Regarding the South Central Railway, an official said, “Most animals that are killed on
the tracks are cattle. The death of the protected animals takes place mostly in states like
Assam, West Bengal and Jharkhand where the track passes through thick forests.
Although drivers reduce the speed in animal sensitive areas, but it’s difficult if an animal
suddenly comes in front of the track.

• Our model detects MONKEY crossing Railway tracks and then can inform
the respective officials about it and then the officials can take actions against that can
save the lives of unsuspecting animals.
## JETSON NANO COMPATIBILITY

• The power of modern AI is now available for makers, learners, and embedded
developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs
in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used
Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end
accelerated AI applications. All Jetson modules and developer kits are supported by
JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release
and supports all Jetson modules.

## PROPOSED SYSTEM
1. Study basics of machine learning and image recognition.

2. Start with implementation
• Front-end development
• Back-end development

3. Testing, analyzing and improvising the model. An application using python and Roboflow
and its machine learning libraries will be using machine learning to identify an animal
when it is crossing Railway tracks.

4. Use data sets to interpret MONKEY and convey it when they are crossing
tracks in the viewfinder.

## METHODOLOGY
The MONKEY detection model on Railway tracks is a program that focuses
on implementing real time MONKEY detection on Railway tracks.
It is a prototype of a new product that comprises of the main module:
MONKEY detection and then showing on viewfinder when one is crossing
tracks according to data fed.
MONKEY Detection Module

This Module is divided into two parts:
1. MONKEY  Detection

• Ability to detect the location of an animal in any input image or frame. The output is the
bounding box coordinates on the detected animal.

• For this task, initially the Data set library Kaggle was considered. But integrating it was a
complex task so then we just downloaded the images from google images and made our
own data set.

• This Data set identifies MONKEY in a Bitmap graphic object and returns the
bounding box image with annotation of name present.

2. Classification Detection

• Classification of the MONKEY based on when they are crossing Railway
tracks on the viewfinder.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision
was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in
production. Given it is natively implemented in PyTorch (rather than Darknet), modifying
the architecture and exporting and deployment to many environments is straightforward.
##bINSTALLATION

sudo apt-get remove –purge libreoffice*

sudo apt-get remove –purge thunderbird*

sudo fallocate -l 10.0G /swapfile1

sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab

#################add line###########

/swapfile1 swap defaults 0 0

vim ~/.bashrc

#############add line #############

exportPATH=/usr/local/cuda/bin${PATH:+:${PATH}}

exportLD_LIBRARY_PATh=/usr/local/cuda/lib64${LD\_LIBRARY\_PATH:+:${LD_LIBRA

RY_PATH}}

exportLD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

sudo apt-get update


sudo apt-get upgrade

################pip-21.3.1 setuptools-59.6.0 wheel-

0.37.1#############################

sudo apt install curl

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo python3 get-pip.py

sudo apt-get install libopenblas-base libopenmpi-dev


sudo apt-get install python3-dev build-essential autoconf libtool pkg-config

python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7

qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script

libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-

dev libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev

libffi-dev libfreetype6-dev python3-dev

vim ~/.bashrc

####################### add line ####################

exportOPENBLAS_CORETYPE=ARMV8



source~/.bashrc

sudo pip3 install pillow

curl -LO

https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-

linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo python3 -c “import torch; print(torch.cuda.is_available())”

git clone –branch v0.9.1 https://github.com/pytorch/vision torchvision

cdtorchvision/

sudo python3 setup.py install

cd

git clone https://github.com/ultralytics/yolov5.git

cdyolov5/

sudo pip3 install numpy==1.19.4

history

#####################comment torch,PyYAML and torchvision in

requirement.txt##################################

sudo pip3 install –ignore-installed PyYAML&gt;=5.3.1

sudo pip3 install -r requirements.txt

sudo python3 detect.py

sudo python3 detect.py –weights yolov5s.pt –source 0

#############################################Tensorflow################

######################################

sudo apt-get install python3.6-dev libmysqlclient-dev

sudo apt install -y python3-pip libjpeg-dev libcanberra-gtk-module libcanberra-

gtk3-module

pip3 install tqdm cython pycocotools

#############

https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorf

low-2.5.0%2Bnv21.8-cp36-cp36m-linux_aarch64.whl ######

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip

libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip

sudo pip3 install -U pip testresources setuptools==49.6.0

sudo pip3 install -U –no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5

keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf

pybind11 cython pkgconfig

sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0

sudo pip3 install -U cython

sudo apt install python3-h5py



sudo pip3 install #install downloaded tensorflow(sudo pip3 install –pre –extra-

index-url https://developer.download.nvidia.com/compute/redist/jp/v46

tensorflow)

python3

import tensorflow as tf

tf.config.list_physical_devices(“GPU”)

print(tf.reduce_sum(tf.random.normal([1000,1000])))

#######################################mediapipe######################

####################################

git clone https://github.com/PINTO0309/mediapipe-bin

ls

cdmediapipe-bin/

ls

./v0.8.5/numpy119x/mediapipe-0.8.5_cuda102-cp36-cp36m-

linux_aarch64_numpy119x_jetsonnano_L4T32.5.1_download.sh

ls

sudo pip3 install mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl

## Demo

https://youtu.be/NCEjeI3_Pxo


ADVANTAGES

• Deaths of animals due to crossing of railway tracks is major cause of concern in India
and around the world our model can be used to mitigate this problem by keeping a
watchful eye on the Railway tracks.

• MONKEY detection system shows MONKEY crossing Railway
tracks in viewfinder of camera module with good accuracy.

• Our model can be used in places where there is less workforce with respect to overall
accident occurences and hence makes the process of recognizing MONKEY 
on tracks more efficient.

• MONKEY detection on Railway tracks model works completely automated
and no user input is required.

• It can work around the clock and therefore becomes more cost efficient.
APPLICATION

• Detects MONKEY and then checks whether they are crossing the Railway
tracks in each image frame or viewfinder using a camera module.

• Can be used anywhere Railway tracks are laid and also places where illegal MONKEY
Railway tracks crossing is regularly observed.

• Can be used as a reference for other ai models based on MONKEY detection
on Railway tracks.

## FUTURE SCOPE
• As we know technology is marching towards automation, so this project is one of the
step towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater
number of epochs.

• MONKEY detection on Railway tracks model will become a necessity in the
future due to the increase in number of trains running on railway tracks and hence our
model will be of great help to tackle the situation in an efficient way. As urbanization and
deforestation increase more animals will leave the jungle and hence more MONKEY would roam free in cities and hence increase in chance of Railway tracks crossing.

## CONCLUSION
• In this project our model is trying to detect a MONKEY and then showing it on
viewfinder, live as to whether they are crossing a Railway track as we have specified in
Roboflow.

• This model tries to solve the problem of animals crossing Railway tracks illegally and
thus reduce the chances of their deaths and also any other consequent accident related
to Railway tracks crossing.

• The model is efficient and highly accurate and hence works without any lag and also if
the data is downloaded can be made to work offline.

## REFERENCE
1. Roboflow:-https://roboflow.com/

2. Datasets or images used :- Google images

## ARTICLES
1. https://www.thehindu.com/news/national/over-32000-animals-killed-on-railway-tracks-in-
2016-18/article28280406.ece?homepage=true

2. https://www.deccanchronicle.com/nation/current-affairs/240719/hyderabad-31-animals-
crushed-to-death-on-rail-tracks-every-day.html
