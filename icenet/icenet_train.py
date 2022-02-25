import os
import torch
import torch.nn as nn  # nn은 레이어 안에서 weight 공유 가능
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from icenet import icenet_model as model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 지금 0 밖에 없음
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "is available")

# 하이퍼파라미터
points = 10  # scribble size
epochs = 50
batch_size = 8
learning_rate = 0.001  # 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.

# 데이터 전처리
trans = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


# 데이터셋 로드
data_filename = '../data/icenet_data/Dataset_Part1.rar'  # .rar: 무손실 압축 포맷
train_set = torchvision.datasets.ImageFolder(root='./data/icenet_data', transform=trans)

# 데이터 로더
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

# loss
class Libc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, y_pred):
        ctx.save_for_backward(y, y_pred)
        return (y_pred - y).pow(2).sum()

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred = ctx.saved_tensors
        grad_input = torch.neg(2.0 * (yy_pred - yy))
        return grad_input, grad_output






model = model.IceNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 이거 말고 다른 거!
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 결과 출력
        running_loss += loss.item()
        if i % 2500 == 2499:  # 50000개의 train 이미지 / batch_size = 2500개마다 print
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2500:.3f}')
            running_loss = 0.0

print('Finished Training')


PATH = './data/icenet_bytrain.pt'  # 저장할 위치, 파일명(폴더는 이미 존재하고 있어야 함)
torch.save(model.state_dict(), PATH)  # 모델 객체의 state_dict 저장하기
# model.load_state_dict(torch.load(PATH))
# model.eval()  # eval()가 없으면 일관성 없는 추론 결과

