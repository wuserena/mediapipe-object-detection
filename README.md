# SEPTA-object-detection

## System Environment  

- **Operating System**: WSL Ubuntu 24.04.1 LTS
- **GPU**: NVIDIA Geforce RTX 4070 Laptop
- **CUDA Version**: 12.7
- **Python Version**: python3.11
- **Tensorflow**: 2.18.0 

* MiniPupper (Raspberry Pi Compute Module 4)  
  - **Camera:**

## Environment Setup

Build a virual environment to mangae the project
```sh
sudo apt install python3.11 python3.11-pip
sudo apt update

python3.11 -m venv mediapipe
source mediapipe/bin/activate
```

Install dependencies using:  

```sh
pip install -r requirements.txt
```

##  Usage
### Create dataset
Selected dataset from video with in 1 frame for every 0.5 seconds

```sh
python ConvertVideoToImage.py
```

After sampled the image, used Roboflow to label.

### Training customer object detection model

The code is retrived from [Object detection model customization guide](https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector)

For training model run:

```sh
python train.py
```

### Apply object detect on image

Object detected by using the customer training model

```sh
python testing.py
```

### Apply object detect on video

Object detected by using the customer training model

```sh
python object_tracking.py
```
### Apply object detect on stream video

Using Raspberry pi camera to capture the stream video. Run this on MiniPupper to start sever API:

```sh
python camera_sever_api.py
```

Run this on PC to process the stream video to active object detection:

```sh
python camera_client.py
```


