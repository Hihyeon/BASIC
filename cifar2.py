# model
import torch
import torch.nn as nn  # nn은 레이어 안에서 weight 공유 가능
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

# dataset and transformation
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# utils
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 학습을 위한 장치 얻기. 가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습. torch.cuda를 사용할 수 있는지 확인하고 그렇지 않으면 CPU를 계속 사용합니다.
torch.manual_seed(113)  # 현재 실습하고 있는 파이썬 코드를 재실행팅해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
if device == 'cuda':
    torch.cuda.manual_seed_all(113)
print(device + " is available")

# 하이퍼파라미터(모델 최적화 과정을 제어할 수 있는 조절 가능 매개변수)
epochs = 5  # 데이터셋을 반복하는 횟수
batch_size = 4  # 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
learning_rate = 0.01  # 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.

# CIFAR-10 데이터셋 로드
train_set = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)
validation_set = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

# 데이터 로더
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True)  # shuffle=True는 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다.
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = models.densenet161().to(device)
# model = models.vgg16().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 리스트에 저장
running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []


# train
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    validation_running_loss = 0.0
    validation_running_correct = 0.0

    for i, data in enumerate(train_loader, 0):  # 0부터 시작
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 결과 출력
        preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels.data)
        running_loss += loss.item()

    # 훈련할 필요가 없으므로 메모리 절약
    else:
        with torch.no_grad():
            for val_input, val_label in validation_loader:

                val_input = val_input.to(device)
                val_label = val_label.to(device)
                val_outputs = model(val_input)
                val_loss = criterion(val_outputs, val_label)

                val_preds = torch.max(val_outputs, 1)
                validation_running_loss += val_loss.item()
                validation_running_correct += torch.sum(val_preds == val_label.data)

                epoch_loss = running_loss / len(train_loader)
                epoch_acc = running_correct.float() / len(train_loader)
                running_loss_history.append(epoch_loss)
                running_correct_history.append(epoch_acc)

                val_epoch_loss = validation_running_loss / len(validation_loader)
                val_epoch_acc = validation_running_correct.float() / len(validation_loader)
                validation_running_loss_history.append(val_epoch_loss)
                validation_running_correct_history.append(val_epoch_acc)

        print("===================================================")
        print("epoch: ", epoch + 1)
        print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
        print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))

# 학습 모델 저장
# PATH = "./data/"  # 저장할 위치, 파일명(폴더는 이미 존재하고 있어야 함)
# torch.save(model.state_dict(), 'model_weights.pt')  # torch.save(model.state_dict(), PATH)  # 모델 객체의 state_dict 저장하기
# # model.load_state_dict(torch.load('model_weights.pt'))
# model.eval()  # eval()가 없으면 일관성 없는 추론 결과