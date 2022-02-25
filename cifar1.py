# model
import cv2
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
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 학습을 위한 장치 얻기. 가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습. torch.cuda를 사용할 수 있는지 확인하고 그렇지 않으면 CPU를 계속 사용합니다.
torch.manual_seed(113)  # 현재 실습하고 있는 파이썬 코드를 재실행팅해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
if device == 'cuda':
    torch.cuda.manual_seed_all(113)
print(device + " is available")

# 하이퍼파라미터(모델 최적화 과정을 제어할 수 있는 조절 가능 매개변수)
epochs = 5  # 데이터셋을 반복하는 횟수
batch_size = 4  # 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
learning_rate = 0.001  # 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.


# CIFAR-10 데이터셋 로드
train_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
test_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

# 데이터 로더
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)  # shuffle=True는 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # 첫번째층: 합성곱층
        # ImgIn shape=(32, 32, 3)
        #    Conv     -> (32, 32, 64)
        #    Pool     -> (16, 16, 64)  (풀링은 가중치가 없어 연산 후 채널 수가 변하지 않음)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 두번째층: 합성곱층
        # ImgIn shape=(16, 16, 64)
        #    Conv     -> (16, 16, 128)
        #    Pool     -> (8, 8, 128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 세번째층: 합성곱층
        # ImgIn shape=(8, 8, 128)
        #    Conv     -> (8, 8, 256)
        #    Pool     -> (4, 4, 256)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 네번째층: 합성곱층
        # ImgIn shape=(4, 4, 256)
        #    Conv     -> (4, 4, 512)
        #    Pool     -> (2, 2, 512)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.avg_pool = nn.AvgPool1d(2)
        self.fc1 = nn.Sequential(  # fc <=> linear
            nn.Linear(2 * 2 * 512, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True),  # 0보다 작은 값은 ReLU가 없애버림
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)  # 전결합층을 위해서 Flatten
        out = self.fc1(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = VGG().to(device)  # 정확도 10%..
# model = Net().to(device)
# model = models.vgg16().to(device)  # 오래 걸림
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# train
# for epoch in range(epochs):
#     running_loss = 0.0
#
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # 결과 출력
#         running_loss += loss.item()
#         if i % 2500 == 2499:  # 50000개의 train 이미지 / batch_size = 2500개마다 print
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2500:.3f}')
#             running_loss = 0.0
#
# print('Finished Training')


PATH = '../data/model_weights.pt'  # 저장할 위치, 파일명(폴더는 이미 존재하고 있어야 함)
# torch.save(model.state_dict(), PATH)  # 모델 객체의 state_dict 저장하기
model.load_state_dict(torch.load(PATH))
model.eval()  # eval()가 없으면 일관성 없는 추론 결과


# test
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # outputs are energies for the 10 classes.
        _, predictions = torch.max(outputs, 1)

        for image, label, prediction in zip(images, labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1

            # test 틀렸을 때 이미지 띄우기
            else:
                print("예상:{}, 정답: {} => 오답!".format(classes[prediction], classes[label]))
                img = np.transpose(torchvision.utils.make_grid(image).cpu().numpy(), (1, 2, 0))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for {classname:5s}: {accuracy:.1f} %')


