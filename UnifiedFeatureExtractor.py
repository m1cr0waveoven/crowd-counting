import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from dataclasses import dataclass
import numpy as np
from deep_person_reid.torchreid.utils import FeatureExtractor

@dataclass(slots=True, frozen=True)
class Person:
    track_id: int
    face_feature: torch.tensor
    body_feature: torch.tensor

class UnifiedFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.deep_feature_extractor: FeatureExtractor = FeatureExtractor(
            model_name = 'osnet_ain_x1_0',
            device = str(self.device)
        )

    @staticmethod
    def draw_face_box(frame: cv2.typing.MatLike, boxes) -> cv2.typing.MatLike:
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (187,128,53), 2, 8)
    
    def extract_face_features(self, image: Image) -> torch.tensor:
        # cv2.imshow('img', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        boxes, _ = self.mtcnn.detect(image)
        if boxes is None:
            return
        # Draw faces
        self.__class__.draw_face_box(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), boxes)
        # print(f"{boxes[0]=}")
        normalized_img, prob = self.mtcnn(image, return_prob=True)
        if normalized_img is None:
            return
        
        embedings = self.resnet(normalized_img.unsqueeze(0).to(self.device))
        return embedings
    
    def extract_person_features(self, frame: cv2.typing.MatLike) -> torch.tensor:
        person_feature = torch.tensor([])        
        person_feature = self.deep_feature_extractor(frame)
        # print(f"{person_feature=}")
        return person_feature

    def extract_features(self, frame: cv2.typing.MatLike, track_id: int) -> Person:
        image: Image = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        face_embedings = self.extract_face_features(image)
        person_feature = self.extract_person_features(frame)
        person = Person(track_id, face_embedings, person_feature)
        return person