# Import necessary libraries
import cv2
import sys
import numpy as np

def detect_faces(image_path):
    # Load pre-trained model from disk
    model_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
    
    # Read the input image
    image = cv2.imread(image_path)
    (image_height, image_width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections and predictions
    model_net.setInput(blob)
    detections = model_net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([image_width, image_height, image_width, image_height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            
            # Draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y_coordinate = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(image, text, (start_x, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # Show the output image
    cv2.imshow("Output", image)
    # Close windows with Esc key
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # Load the model and detect faces in the provided image
        detect_faces(image_path = sys.argv[1])
    except IndexError:
        # Print usage message if no image path provided
        print(f"Usage: {sys.argv[0]} image_path")
        print("Error: Image not found")
