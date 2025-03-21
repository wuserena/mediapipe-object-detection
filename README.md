# SEPTA-object-detection

## System Environment  

- **Operating System**: WSL Ubuntu 24.04.1 LTS
- **GPU**: NVIDIA Geforce RTX 4070 Laptop
- **CUDA Version**: 12.7
- **Python Version**: python3.11
- **Tensorflow**: 2.18.0 

* MiniPupper (Raspberry Pi Compute Module 4)  
  - **Screen:**  Displays the chatbot's responses.
  - **Servo:** Enables tracking of the user.
  - **Camera:** Used for face detection to track the user.
  - **Microphone:** (on MiniPupper) Captures speech for recognition.
  - **Speaker:** (on the screen) Outputs the chatbot’s responses for the user to hear.

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


## Requirements

Install them using:

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

### Training custom object detection model customization

The code is retrived from [Object detection model customization guide](https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector)

For training model run:

```sh
python train.py
```


