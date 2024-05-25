# crowd-counting
Contains code related to my diploma work.

Folders:
  YOLOv8_benchmark: contains script the was used to measure the performance of different expoter models
  
  face-detection: contains code that was initialy developed to detect faces on video.
  
  faceverification: contains a the script file used for faceverification using FaceNet, the script outputs the disctance between each faces
  
  transfer python object over TCP: contains two scripts client and server. Simple demonstration, how to send a complex Python object through a TCP connnection.

  CrowdCounting.py: the Python script that count people, the entry point of the application. Uses UnifiedFeatureExtractor.py

  UnifiedFeatureExtractor.py: unifies the facial feature embeding extration (FaceNet) with deep feature extraction for person re-identification.

  re-id.py: used for experimnets with preson re-identification using Torchreid.

References:

ultralytics/ultralytics: [YOLOv8](https://github.com/ultralytics/ultralytics)

timesler/facenet-pytorch: [facenet-pytorch](https://github.com/timesler/facenet-pytorch)

KaiyangZhou/deep-person-reid: [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)

opencv/opencv-python: [OpenCV Python](https://github.com/opencv/opencv-python)





