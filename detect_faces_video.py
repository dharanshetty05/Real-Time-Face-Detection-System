# Import necessary libraries
import cv2
import time
import imutils
import numpy as np
from imutils.video import VideoStream

# Load pre-trained model from disk and initialize the video stream
model_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') 
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the video stream and resize it to have a maximum width of 400 pixels
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)
 
    # Grab the frame dimensions and convert it to a blob
    (frame_height, frame_width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
    # Pass the blob through the network and obtain the detections and predictions
    model_net.setInput(blob)
    detections = model_net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        # You can also adjust the threshold (0.5 here) for better results
        if confidence < 0.5:
            continue

        # Compute the (x, y) coordinates of the bounding box for the object
        bounding_box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
        (start_x, start_y, end_x, end_y) = bounding_box.astype("int")
 
        # Draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y_coordinate = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(frame, text, (start_x, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    # Close windows with Esc key
    key = cv2.waitKey(1) & 0xFF
 
    # Break the loop if ESC key is pressed
    if key == 27:
        break

# Destroy all the windows
cv2.destroyAllWindows()
video_stream.stop()
