import numpy as np
import cv2

def onChange(pos):
    pass

def trackbar():
    img = np.zeros((200, 512, 3), np.uint8)  # 200x512 크기의 검정색 그림판 생성
    cv2.namedWindow(('color_palette'))

    cv2.createTrackbar('R', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('G', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('B', 'color_palette', 0, 255, onChange)

    switch = '0: OFF\n1: ON\n'
    cv2.createTrackbar(switch, 'color_palette', 0, 1, onChange)

    resume = True
    while (resume):
        cv2.imshow('color_palette', img)
        # 사용자가 키보드를 두드릴 때까지 기다리고 키보드 입력이 되면 그 값을 k로 한다.
        # 0xFF를 &연산 한 이유는 운영체계가 64비트라서(32비트 운영체제는 &0xFF할 필요 없음)
        k = cv2.waitKey(1) & 0xFF

        # To end demo, push key "Esc"
        if k == 27:
            resume = False
            break

        # R, G, B, ON/OFF의 트랙바 현재 값을 r, g, b, s로 한다.
        r = cv2.getTrackbarPos('R', 'color_palette')
        g = cv2.getTrackbarPos('G', 'color_palette')
        b = cv2.getTrackbarPos('B', 'color_palette')
        s = cv2.getTrackbarPos(switch, 'color_palette')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]
    cv2.destroyAllWindows()

trackbar()