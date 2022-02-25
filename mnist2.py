# 기본적인 셋팅
import torch
import torch.nn as nn  # nn은 레이어 안에서 weight 공유 가능

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt  # 시각화를 위한 맷플롯립
import numpy as np
import cv2
import glob
from PIL import Image

np.set_printoptions(formatter=dict(int=lambda x : f'{x:4}'))  # 출력의 정밀도를 설정하는 데 사용

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 학습을 위한 장치 얻기. 가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습. torch.cuda를 사용할 수 있는지 확인하고 그렇지 않으면 CPU를 계속 사용합니다.
torch.manual_seed(77)  # 현재 실습하고 있는 파이썬 코드를 재실행팅해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
if device == 'cuda':
    torch.cuda.manual_seed_all(77)
print(device + " is available")

# 하이퍼파라미터(모델 최적화 과정을 제어할 수 있는 조절 가능 매개변수)
epochs = 10  # 데이터셋을 반복하는 횟수
batch_size = 100  # 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
learning_rate = 0.001  # 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.

# MNIST 데이터셋 로드
train_set = torchvision.datasets.MNIST(
    root = './data/MNIST',  # 학습/테스터 데이터가 저장되는 경로
    train = True,  # 학습용 또는 테스트용 데이터셋 여부 지정
    download = True,  # True 시 root에 데이터가 없는 경우 인터넷에서 다운로드
    # transform과 targe_transform은 특징(feature)과 정답(label) 변형(transform)을 지정
    # transforms. Compose torchvision.transforms는 데이터를 전처리하는 패키지.
    transform = transforms.Compose([
        transforms.ToTensor()  # 데이터 타입을 Tensor 형태로 변경 / 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 하지만 ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜.
    ])
)

test_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

# 데이터 로더
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # shuffle=True는 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# 신경망 모델(3층)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층: 합성곱층
        # ImgIn shape=(28, 28, 1)
        #    Conv     -> (28, 28, 32)
        #    Pool     -> (14, 14, 32)  (풀링은 가중치가 없어 연산 후 채널 수가 변하지 않음)
        # input channel = 1, filter(output channel) = 32, kernel size = 5x5, zero padding = 2, stride = 1
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 두번째층: 합성곱층
        # ImgIn shape=(14, 14, 32)
        #    Conv      ->(14, 14, 64)
        #    Pool      ->(7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 전결합층 7x7x20 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

# 객체, 손실 함수, 최적화 함수 생성
# model = nn.Linear(784, 10, bias=True).to(device)  # 단순히 이미지의 화소값을 특징으로 일렬로 구성, 이미지라는 2차원적 개념 무시 (90%의 정확도)
model = CNN().to(device) # CNN 모델 정의 (98%의 정확도)
criterion = nn.CrossEntropyLoss().to(device)  # nn.CrossEntropyLoss()는 기본적으로 LogSoftmax()가 내
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # model.parameters()를 통해 model의 파라미터들을 할당 / lr : learning_rate 지정

# 총 배치의 수
total_batch = len(train_loader)  # (total / batch_size)
print('총 배치의 수 : {}'.format(total_batch))

# train
for epoch in range(epochs): # epochs수만큼 반복
    avg_cost = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad() # 모든 model의 gradient 값을 0으로 설정
        hypothesis = model(data) # 모델을 forward pass해 결과값 저장
        cost = criterion(hypothesis, target) # output과 target의 loss 계산
        cost.backward() # backward 함수를 호출해 gradient 계산
        optimizer.step() # 모델의 학습 파라미터 갱신
        avg_cost += cost / len(train_loader) # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# test
with torch.no_grad():  # 학습을 진행하지 않을 것이므로 torch.no_grad()
    correct = 0
    total = 0

    title = "TEST"

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        prediction = torch.max(out.data, 1)[1]  # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
        total += len(target)  # 전체 클래스 개수
        correct += (prediction == target).sum().item()  # 예측값과 실제값이 같은지 비교

        img = np.transpose(torchvision.utils.make_grid(data).cpu().numpy(), (1, 2, 0))

    print('Test Accuracy: ', 100. * correct / total, '%')

    print('Labels: ', target.cpu().numpy())
    print('정답: ', correct, '/', total)

    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

