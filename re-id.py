from deep_person_reid.torchreid.utils import FeatureExtractor
import deep_person_reid.torchreid
import cv2
import torch
import typing
import os
import numpy as np
from scipy.spatial import distance
from UnifiedFeatureExtractor import Person, UnifiedFeatureExtractor

# osnet_ain_x0_25_imagenet.pyth
# img = cv2.imread('person.jpg')
# cv2.imshow('img1',img)
# cv2.waitKey(10)

# img1 = cv2.imread('person_crop1.jpg')
# img2 = cv2.imread('person_crop1.jpg')
# img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img_ndarray= [img1_rgb, img2_rgb]

# img_ndarray_gray= []
# # img_ndarray_gray.append(cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY, 3), cv2.COLOR_GRAY2BGR))
# img_ndarray_gray.append(img1_rgb)
# img_ndarray_gray.append(cv2.cvtColor(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY, 3), cv2.COLOR_GRAY2BGR))
# cv2.imshow('img1',img_ndarray_gray[1])
# cv2.waitKey(1000)
# print(f"{img_ndarray_gray[0].ndim=}")
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    # model_name='resnet50mid',
    # model_path='osnet_x1_0_imagenet.pth',
    # model_path = 'resnet50-19c8e357.pth',
    device='cuda'
)

image_list = [
    'person1_45.jpg',
    'person1_50.jpg',
    'person1_55.jpg',
    'person1_61.jpg',
    'person2_5.jpg',
    'person2_10.jpg',
    'person2_15.jpg',
    'person2_20.jpg',
    'person3_5.jpg',
    'person3_10.jpg',
    'person3_15.jpg',
    'person3_20.jpg'
]

distances: list[torch.tensor] = []
features_pil = extractor(image_list)

features_length = len(features_pil)
output_length = int((features_length * (features_length-1))/2)
print(f"{output_length=}")
person_index = 1
person_index2 = 1
for i, feature in enumerate(features_pil):
    for j in range(i+1, features_length):
        dist = torch.dist(feature, features_pil[j])
        # dist = distance.cosine(feature.cpu().numpy(), features_pil[j].cpu().numpy())
        # print(f"{i} {j}")
        # print(f"{i}-{j}.: {dist}")
        print(f"{dist}")
        distances.append(torch.dist(feature, features_pil[j]))

print(f"{len(distances)=}")
# 1 - 2
# 1 - 3
# 1 - 4
# 2 - 3
# 2 - 4
# 3 - 4


def find_matching_persons(person_list: list[Person]):
    matching_persons: list[list[(Person, str)]] = []
    person_lsit_length = len(person_list)
    for i, person1 in enumerate(person_list):
        matching_persons_sub: list[Person] = [(person1, "query")]
        # for j in range(i+1, person_lsit_length):
        for j, person2 in enumerate(person_list):
            # person2 = person_list[j]
            if i == j: 
                continue
            face_distance = None
            body_distance = None
            if person1.face_feature is not None and person2.face_feature is not None:
                face_distance = torch.dist(person1.face_feature, person2.face_feature)
                # print(f"{face_distance=}")
            if person1.body_feature is not None and person2.body_feature is not None:
                body_distance = torch.dist(person1.body_feature, person2.body_feature)
                # print(f"{body_distance=}")
            if face_distance is not None and body_distance is not None:
                if face_distance <= 0.95 and body_distance <= 19:
                    matching_persons_sub.append((person2,"face&body"))
                    # print(f"{body_distance}")
                    print(f"Append based on face and body {person1.track_id} - {person2.track_id}")
                    
            elif face_distance is not None:
                if face_distance <= 0.95:
                    matching_persons_sub.append((person2, "face"))
                    print(f"Append based on face {person1.track_id} - {person2.track_id}")
            elif body_distance is not None:
                if body_distance <= 19:
                    matching_persons_sub.append((person2, "body"))
                    print(f"Append based on body {person1.track_id} - {person2.track_id}")
                    
        if(len(matching_persons_sub) != 0):
            matching_persons.append(matching_persons_sub)
    return matching_persons

unified_extractor = UnifiedFeatureExtractor()
images = []
for path in image_list:
    images.append(cv2.imread(path))
print("----------------------------------- CV2 imread list ------------------------")
for i, image in enumerate(images):
    for j in range(i+1, features_length):
        feature1 = extractor(np.array(image))
        feature2 = extractor(np.array(images[j]))
        dist = torch.dist(feature1, feature2)
        print(f"{dist}")
print("---------------------------------------------------------------------------")

persons: list[Person] = []
for i, img in enumerate(images):
    persons.append(unified_extractor.extract_features(img, i+1))

matching_persons: list[list[Person]] = find_matching_persons(persons)
for i, personlist in enumerate(matching_persons, start=1):
    # print(f"{len(personlist)=}")
    print(f"Similar persons to person the query person {i}:")
    for p in personlist:
        person, similarity_kind = p
        print(f"{person.track_id} - {similarity_kind}")
# features_nd_gray = extractor(img_ndarray_gray)
# features_nd = extractor(img_ndarray)
# print(f"{features=}")
# print(features.shape) # output (5, 512)
# print(f"{torch.dist(features_nd_gray[0], features_nd_gray[1])=}")
# print(f"{torch.dist(features_nd[0], features_nd[1])=}")
# print(f"{features[0]=}")
# print(f"{features[0].dim()=}")
# dist_nd_gray = distance.euclidean(features_nd_gray[0].cpu().numpy(), features_nd_gray[1].cpu().numpy())
# dist_nd = distance.cosine(features_nd[0].cpu().numpy(), features_nd[1].cpu().numpy())
# dist_pil = distance.cosine(features_pil[0].cpu().numpy(), features_pil[1].cpu().numpy())
# print(f"{dist_nd_gray=}")
# print(f"{dist_nd=}")
# print(f"{dist_pil=}")
# torchreid.models.show_avai_models()

