import numpy as np
import cv2
import random

import os
import matplotlib.pyplot as plt

def RANSAC(lines, iterations, threshold):
    best_line1 = None
    best_line2 = None
    best_inliers = 0
    for i in range(iterations):
        idx = random.randint(0, len(lines) - 1)
        line1 = lines[idx][0]
        inliers1 = [idx]
        for j in range(len(lines)):
            if j == idx:
                continue
            line2 = lines[j][0]
            distance = np.abs(line2[0] * line1[1] - line2[1] * line1[0]) / np.sqrt(line1[0] ** 2 + line1[1] ** 2)
            if distance < threshold:
                inliers1.append(j)
        if len(inliers1) > best_inliers:
            best_inliers = len(inliers1)
            best_line1 = line1
            best_line2 = line2
    return best_line1, best_line2


SLASH = '/'

# open the image wiith cv2 and show it
if os.getcwd().split(SLASH)[-1] == 'src':
    os.chdir('..')
path = ''. join([os.getcwd(), SLASH, 'assets', SLASH, 'train', SLASH])

name = 'image'+str(9)+'.png'
path = os.path.join(path, name)

print('path: ', path)

# Using cv2.imread() method to the path 
# but we only want de green 'g' channel
# the others are 0
img = cv2.imread(path, -1)
print('image shape og:', img.shape)

img = img[30:570,30:570,1]

print('img:', img.shape)
print('len shape', len(img.shape))

plt.imshow(img)
plt.show()

gray = img 

# implement ransac for 2 lines detection
edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

line1, line2 = RANSAC(lines, 100, 10)
if line1 is not None and line2 is not None:
    x1, y1, x2, y2 = line1
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    x1, y1, x2, y2 = line2
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Para plotar as retas, precisamos de dois pontos para cada uma
y1 = 0
x1 = (y1 - b1) / m1
y2 = 540
x2 = (y2 - b1) / m1
pt1 = (int(x1), int(y1))
pt2 = (int(x2), int(y2))
cv2.line(img, pt1, pt2, (255, 0, 0), 2)

y3 = 0
x3 = (y3 - b2) / m2
y4 = 540
x4 = (y4 - b2) / m2
pt3 = (int(x3), int(y3))
pt4 = (int(x4), int(y4))
cv2.line(img, pt3, pt4, (0, 255, 0), 2)

# Finalmente, plotamos a imagem com as duas retas
plt.imshow(img)
plt.show()

