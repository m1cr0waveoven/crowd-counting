import cv2
from ultralytics import YOLO
import torch
import numpy as np
from typing import Dict
import pickle
import mysql.connector
from UnifiedFeatureExtractor import UnifiedFeatureExtractor, Person
from numba import njit, jit

registered_persons: Dict[int, Person] = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

top_left_corner = (0,0)
bottom_right_corner = (0,0)
def drawRectangle(action, x, y, flags, *userdata):
        # Referencing global variables 
        global top_left_corner, bottom_right_corner
        # Mark the top left corner when left mouse button is pressed
        if action == cv2.EVENT_LBUTTONDOWN:
            top_left_corner = (x,y)
            print(f"{top_left_corner=}")  
            # When left mouse button is released, mark bottom right corner
        elif action == cv2.EVENT_LBUTTONUP:
            bottom_right_corner = (x,y)
            print(f"{bottom_right_corner=}") 

def main() -> None:
    extractor = UnifiedFeatureExtractor()         
    cv2.namedWindow("YOLOv8 Tracking")
    cv2.setMouseCallback("YOLOv8 Tracking", drawRectangle)
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    # Open the video file
    video_path = "hard_cut1.mp4"
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    # Loop through the video frames
    index = 0
    while success:
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_resized = cv2.resize(frame, (1280, 720))
            cv2.rectangle(frame_resized, top_left_corner, bottom_right_corner, (187,128,53), 2, 8)
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame_resized, persist=True, device=device, classes=0)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # results[0].save_crop()
            if results[0].boxes.id is None:
                continue
            boxes_xywh = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            # confidences = results[0].boxes.conf.cpu().numpy().astype(int)
            # class_id = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, track_id in zip(boxes_xywh, track_ids):
                x, y, _, h = box
                
                center_point = (int(x), int(y+h*0.5))
                cv2.circle(annotated_frame, center_point, radius=5, color=(80, 134, 240), thickness=-1)
                is_in_registration_zone = is_point_inside_rectangle(center_point,top_left_corner, bottom_right_corner)
                # frame_out = crop_boundingbox_from_box_xywh(frame_resized, box)
                # cv2.imwrite(f"person3{index}.jpg", frame_out)
                # index = index + 1
                print(f"{is_in_registration_zone=}")
                                
                if(is_in_registration_zone):
                    x, y, w, h = box
                    x, y, w, h = (int(x),int(y), int(w), int(h))
                    point1, point2 = draw_box_to_frame(annotated_frame, x, y, w, h)
                    # croped_frame = crop_boundingbox(frame, point1, point2)
                    croped_farme = crop_boundingbox_from_xywh(frame_resized, x, y, w, h)
                    if track_id not in registered_persons:
                        person: Person = extractor.extract_features(croped_farme, track_id)
                        registered_persons[track_id] = person
                        print(f"New preson was registered with track_id: {track_id}")
                        insert_personn_to_database(person)
            print(f"Registered persons count: {len(registered_persons)}")
                    
                                           
            # Display the annotated frame           
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()
    
# def process_boxes(boxes_xywh: list[torch.tensor], track_ids: list[int]) -> None:
#     pass
@jit(nopython=False)
def draw_box_to_frame(frame: cv2.typing.MatLike, x: int, y: int, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y, w, h = (int(x), int(y), int(w), int(h))
    point1 = (x + int(w*0.5), y + int(h*0.5))
    point2 = (x - int(w*0.5), y - int(h*0.5))
    # print((x, y, w, h))
    cv2.rectangle(frame, point1, point2, (0, 255, 0), 2)
    return point1, point2

@njit
def crop_boundingbox(frame: cv2.typing.MatLike, point1: tuple[int, int], point2: tuple[int, int]) -> cv2.typing.MatLike:
    crop_frame = frame[point2[1]:point1[1], point2[0]:point1[0]]
    return crop_frame

@njit
def crop_boundingbox_from_xywh(frame: cv2.typing.MatLike, x_center: int, y_center: int, w: int, h: int):
    w, h = (int(w), int(h))
    # Calculate top-left corner coordinates
    x = int(x_center - w*0.5)
    y = int(y_center - h*0.5)
    # Ensure the bounding box is within the image boundaries
    x = max(1, x)
    y = max(1, y)
    x_end = min(frame.shape[1]-1, x+w)
    y_end = min(frame.shape[0]-1, y+h)
    # Extract the region of interest (ROI)
    crop_frame = frame[y:y_end, x:x_end]
    return crop_frame

@njit
def is_point_inside_rectangle(point, top_left_corner, bottom_right_corner):
    (x, y) = point
    (x1, y1) = top_left_corner
    (x2, y2) = bottom_right_corner
    # Check if the point is inside the rectangle
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    return False

def find_matching_persons(person_list: list[Person]):
    matching_persons: list[list[Person]] = []
    person_lsit_length = len(person_list)
    for i, person1 in enumerate(person_list):
        matching_persons_sub: list[Person] = [person1]
        for j in range(i+1, person_lsit_length):
            person2 = person_list[j]
            face_distance = torch.dist(person1.face_feature, person2.face_feature)
            body_distance = torch.dist(person1.body_feature, person2.body_feature)
            if face_distance <= 0.95 and body_distance <= 19:
                matching_persons_sub.append(person1)
                matching_persons_sub.append(person2)

        if(len(matching_persons_sub) != 0):
            matching_persons.append(matching_persons_sub)

    return matching_persons


def connect_to_database():
    # Connect to the MySQL database
    db = mysql.connector.connect(
    host="localhost",
    user="username",
    password="password",
    database="database"
    )
    cursor = db.cursor()
    return cursor

def insert_personn_to_database(person: Person, cursor):
    pickled_person: bytes = pickle.dumps(person)
    query = "INSERT INTO visitors (person_bytes, time) VALUES (%s, NOW())" # person_bytes must be BLOB
    cursor.execute(query, (pickled_person,))

if __name__ == "__main__":
    main()

# CREATE TABLE `visitor_db`.`areas` (`id` INT NOT NULL AUTO_INCREMENT , `name` VARCHAR(30) NOT NULL , PRIMARY KEY (`id`)) ENGINE = InnoDB; 
# CREATE TABLE `visitor_db`.`visitors` (`id` INT NOT NULL AUTO_INCREMENT , `person_bytes` BLOB NOT NULL , `area_id` INT NOT NULL , PRIMARY KEY (`id`)) ENGINE = InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_hungarian_ci; 
