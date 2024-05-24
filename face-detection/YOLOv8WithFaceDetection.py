from collections import defaultdict

import cv2
import numpy as np
from time import time
from ultralytics import YOLO

import torch

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU acceleration.")
else:
    print("CUDA is not available. PyTorch will use CPU for computations.")
    
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "medium2.mp4"
video_out="medium2_face_processed_BOT-SORT.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap_out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame.shape[1], frame.shape[0]))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default_CUDA.xml')
start_time = 0
end_time = 0
# Store the track history
track_history = defaultdict(lambda: [])
fpslist=[]

# Loop through the video frames
while cap.isOpened():
    start_time = time()
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.8, iou=0.6, persist=True, classes=0, device=0, agnostic_nms=True)
        if results[0].boxes.id == None:
            continue 
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        if not 0 in class_id:
            continue
        #print(f"class ids: {class_id}")
        #print(f"tracker ids:{len(track_ids)}")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            # if len(track) > 90:  # retain 90 tracks for 90 frames
              #  track.pop(0)
            
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

        end_time = time()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cap_out.write(annotated_frame)
        fps = 1 / np.round(end_time - start_time, 2)
        fpslist.append(fps)

        print(f'FPS: {int(fps)}')
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
FPS = sum(fpslist) / len(fpslist)
print(FPS)
print(f"Track id's count: {len(track_history.keys())}")
cap.release()
cap_out.release()
cv2.destroyAllWindows()
