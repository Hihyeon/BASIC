import os
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2ycbcr

from icenet import icenet_model as model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 지금 0 밖에 없음
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "is available")
points = 10  # scribble size

# initialize
l_drawing, r_drawing = False, False

def onChange(pos):
    pass

def draw_circle(event,x,y,flags,param):
    global l_drawing, r_drawing

    if event == cv2.EVENT_RBUTTONDOWN:
        r_drawing = True
        l_drawing = False

    elif event == cv2.EVENT_LBUTTONDOWN:
        l_drawing = True
        r_drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if l_drawing == True:
            cv2.circle(inputs, (x,y), 5, (0,0,255), -1)
            cv2.circle(scribble, (x,y), points, 1, -1)
        elif r_drawing == True:
            cv2.circle(inputs, (x,y), 5, (255,0,0), -1)
            cv2.circle(scribble, (x,y), points, -1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        l_drawing = False
        cv2.circle(inputs, (x,y), 5, (0,0,255), -1)
        cv2.circle(scribble, (x,y), points, 1, -1)

    elif event == cv2.EVENT_RBUTTONUP:
        r_drawing = False
        cv2.circle(inputs, (x,y), 5, (255,0,0), -1)
        cv2.circle(scribble, (x,y), points, -1, -1)

# load image
img = Image.open('../data/test/LightHouse.png')
img = np.asarray(img)

# rgb2y -> Tensor
ycbcr = rgb2ycbcr(img)  # rgb2y -> Tensor
y = ycbcr[..., 0] / 255.  # 점이 왜 이렇게 많을까...........
y = torch.from_numpy(y).float()
y = y[None, None].to(device)

# rgb -> Tensor
lowlight = torch.from_numpy(img).float()
lowlight = lowlight.permute(2, 0, 1)
lowlight = lowlight.to(device).unsqueeze(0) / 255.

IceNet = model.IceNet().to(device)
IceNet.load_state_dict(torch.load('icenet.pth'))

resume = True

while (resume):
    inputs = img.copy() / 255.
    scribble = np.zeros(inputs.shape[:2])

    drawing = False

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('image', draw_circle)
    cv2.createTrackbar("threshold", "image", 0, 100, onChange)
    cv2.setTrackbarPos("threshold", "image", 60)
    while (1):
        global_e = cv2.getTrackbarPos("threshold", "image") / 100.
        # annotations
        s = torch.from_numpy(scribble)[None, None].float().to(device)
        eta = torch.Tensor([global_e]).float().to(device)
        # feedforward
        enhanced_image = IceNet(y, s, eta, lowlight)
        output = enhanced_image[0].permute(1, 2, 0).cpu().detach().numpy()

        cv2.imshow('image', np.concatenate([inputs, output], 1)[..., ::-1])
        k = cv2.waitKey(1) & 0xFF
        # To reset , push key "1"
        if k == 49:
            resume = True
            break
        # To save results, push key "2"
        if k == 50:
            cv2.imwrite('results/eta_%02d.png' % (global_e * 100), output[..., ::-1] * 255.)
        # To end demo, push key "Esc"
        if k == 27:
            resume = False
            break
    cv2.destroyAllWindows()