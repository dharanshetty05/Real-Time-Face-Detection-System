# Face Detection Project

This project utilizes computer vision techniques to detect human faces in images or live video streams. It employs a pre-trained deep learning model to accurately identify and localize faces within the input data.

## Technologies Used

- OpenCV: Open Source Computer Vision Library
- Python: Programming language used for implementation
- NumPy: Library for numerical computations
- imutils: A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, etc., easier
- Caffe Model: Convolutional Architecture for Fast Feature Embedding (CAFFE) model used for face detection

## Models

The project utilizes the following pre-trained model for face detection:
- Caffe Model: The model is based on the Single Shot MultiBox Detector (SSD) framework and is trained on the Caffe deep learning framework. It detects faces with high accuracy and speed.

## How to Run

To run the face detection script:
1. Install the required dependencies using pip:
    ```bash
    pip install opencv-python imutils numpy
    ```
2. Download the Caffe model files:
    - `deploy.prototxt.txt`
    - `res10_300x300_ssd_iter_140000.caffemodel`
3. Place the downloaded model files in the same directory as the Python scripts.
4. Run the script by executing the following command in the terminal in the same directory as the Python scripts:
For images:
    ```bash
    python detect_faces_image.py 
    ```
    
    OR
    For live stream/videos:    
    ```bash
    python detect_faces_video.py 
    ```
    
    
## License

This project is licensed under the [MIT License](LICENSE).
