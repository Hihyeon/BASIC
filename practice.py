import os
import torch
import torch.nn as nn  # nn은 레이어 안에서 weight 공유 가능
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2ycbcr


# 컬러 이미지 대비 조정
# 하이퍼파라미터
points = 10  # scribble size
epochs = 50
batch_size = 8
learning_rate = 0.001

# 데이터 전처리
trans = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 데이터셋 로드
# data_filename = './data/icenet_data/Dataset_Part1.rar'  # .rar: 무손실 압축 포맷
# train_set = torchvision.datasets.ImageFolder(root='./data/icenet_data', transform=trans)

# 데이터 로더
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
PATH = '../data/test/LightHouse.png'

# 이미지 읽기
src = cv2.imread(PATH, cv2.IMREAD_COLOR)
# bgr 색공간 이미지를 lab 색공간 이미지로 변환
lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
# l, a, b 채널 분리
l, a, b = cv2.split(lab)
# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
# CLAHE 객체에 l 채널 입력하여 CLAHE가 적용된 l 채널 생성
l = clahe.apply(l)

# 정답 이미지 만들기

#
# # l, a, b 채널 병합
# lab = cv2.merge((l, a, b))
# # lab 색공간 이미지를 bgr 색공간 이미지로 변환
# cont_dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
# # 원본, 대비 증가 이미지 화면 출력
# cv2.imshow('org', src)
# cv2.imshow('Increased contrast', cont_dst)
#
# # 화면 출력창 대기/닫기
# cv2.waitKey()
# cv2.destroyAllWindows()
