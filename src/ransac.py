import numpy as np
import cv2
import random

import os
import matplotlib.pyplot as plt

def RANSAC(img):
    # Converter a imagem para binário utilizando um threshold de limiarização
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Encontrar os contornos da imagem binária
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Selecionar os dois maiores contornos
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    # Inicializar as equações das retas
    lines = []
    for contour in contours:
        # Converter o contorno em uma matriz de pontos
        points = contour.reshape(-1, 2)
        
        # Aplicar RANSAC para encontrar a equação da reta
        model = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = model

        m = vy/vx
        b = y0 - m * x0
        lines.append((m, b))
    
    # Devolver os coeficientes das equações das retas
    # escrever em termos de m e b
    m1, m2, b1, b2 = lines[0][0], lines[1][0], lines[0][1], lines[1][1]
    return m1, m2, b1, b2

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

img = img[30:570,30:570]

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('img:', img.shape)
print('len shape', len(img.shape))

#plt.imshow(img)
#plt.show()

# Aplicar RANSAC para encontrar as equações das retas
m1, m2, b1, b2 = RANSAC(img)

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

