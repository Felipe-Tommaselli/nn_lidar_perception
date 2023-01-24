import cv2
import os
import matplotlib.pyplot as plt
# path
path = os.getcwd()
name = 'image0.png'
path = os.path.join(path, name)

# Using cv2.imread() method
# Using 0 to read image in grayscale mode
img = cv2.imread(path, -1)

print('hello ')

plt.imshow(img)
plt.show()
