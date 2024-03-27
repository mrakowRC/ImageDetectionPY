import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import cv2
import requests
import json
from datetime import datetime
import os
import time

# Define or obtain location information
# For example, a static location could be defined as follows:
location_info = {
    "latitude": 40.712776,
    "longitude": -74.005974,
    "description": "New York, NY, USA"
}

# Directory paths
images_dir = 'images'
metadata_dir = 'metadata'

# Create directories if they don't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

if not os.path.exists(metadata_dir):
    os.makedirs(metadata_dir)

def fetch_frame(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert PIL Image to an array compatible with OpenCV
    #return img

def save_image(image, frame_count):
    # Get the current timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_identifier = f"{frame_count}_{timestamp}"
    frame_path = os.path.join(images_dir, f"frame_{unique_identifier}.jpg")
    cv2.imwrite(frame_path, image)
    return unique_identifier  # Return the unique identifier used in the filename

# Function to save metadata to a JSON file
def save_metadata(metadata, unique_identifier):
    metadata_path = os.path.join(metadata_dir, f"frame_{unique_identifier}_metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

camera_feed_url = "http://10.1.1.62/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user=admin&password=ford1000"  # URL to fetch the live feed frame
#cap = cv2.VideoCapture(camera_feed_url)

model = YOLO('yolov8n.pt') # pass any model type

# Train the model on the Open Images V7 dataset
results = model.train(data='open-images-v7.yaml', epochs=100, imgsz=640)

frame_count = 0

#while cap.isOpened():
while True:
    # Read a frame from the video
    frame = fetch_frame(camera_feed_url)
    #success, frame = cap.read()

    #if success:
    if frame is not None:
        # Run YOLOv8 inference on the frame
        resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # Visualize the results on the cropped frame
        results = model(resized_frame, conf=0.1)

        # Combine the original frame with the annotated detections
        annotated_frame = results[0].plot()

        # Save the annotated frame using the save_image function
        unique_identifier = save_image(annotated_frame, frame_count)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        num_objects = len(results[0].boxes)

        # If there are two or more objects, save the frame and print a statement
        if num_objects >= 0:
            # Get the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Extract metadata for saving
            metadata = {
                "num_objects": num_objects,
                "timestamp": current_time,
                "location": location_info,
                "image_path": os.path.join(images_dir, f"frame_{unique_identifier}.jpg")  # Construct the path using the unique identifier
            }
           
            print(metadata)
        
        
        # Save metadata with consistent naming
        save_metadata(metadata, unique_identifier)
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Optional: Add a short delay to control the rate of frame processing
        time.sleep(0.5)  # Adjust the sleep time as needed
       
    else:
        print("Failed to fetch frame.")
        break  # Exit the loop if unable to fetch a new frame

# Release the video capture object and close the display window
#cap.release()
cv2.destroyAllWindows()
