<!-- Technology Stack Badges -->
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Model-green)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white)
![Imutils](https://img.shields.io/badge/Imutils-Utility-orange)

Face Mask Detection System

A deep learning-powered application to detect whether a person is wearing a face mask in real-time using image or video streams. The system uses a fine-tuned MobileNetV2 model, built with TensorFlow and OpenCV, and supports both image-based and video-based detection.
# Features

- Real-time face mask detection from webcam or image input
- Pre-trained using **MobileNetV2** on a labeled mask/no-mask dataset
- Deployment-ready with OpenCV for inference
- Modular scripts for training, image prediction, and live video stream detection
- Easy-to-use CLI arguments for configuration

# Tech Stack

- Model: MobileNetV2 (TensorFlow / Keras)
- Frameworks: OpenCV, NumPy, Matplotlib, Imutils
- Scripts:
  - `train_mask_detector.py` – model training  
  - `detect_mask_image.py` – predict mask in a static image  
  - `detect_mask_video.py` – detect masks in real-time webcam stream  

# Project Structure

├── dataset/ # Dataset of masked and unmasked faces
├── mask_detector.model # Trained model (HDF5)
├── detect_mask_image.py # Script to run inference on images
├── detect_mask_video.py # Script for webcam-based detection
├── train_mask_detector.py # Training script
├── plot.png # Training loss/accuracy graph
├── README.md


# How to Run the Project

1. Clone the repository
   Open your terminal or command prompt and run:

   ```
   git clone https://github.com/yourusername/face-mask-detector.git
   cd face-mask-detector
   ```

2. Install the required dependencies
   Make sure you have Python installed (Python 3.6+ is recommended). Then run:

   ```
   pip install -r requirements.txt
   ```

3. Train the model (optional
   If you want to retrain the model using your own dataset:

   ```
   python train_mask_detector.py --dataset dataset
   ```

4. Run mask detection on an image
   To detect a mask in a static image:

   ```
   python detect_mask_image.py --image path/to/image.jpg
   ```

5. Run real-time mask detection using webcam
   This will open your webcam and start detecting masks in real-time:

   ```
   python detect_mask_video.py
   ```



