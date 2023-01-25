import cv2
import os
import matplotlib.pyplot as plt
# path
path = os.getcwd()
name = 'image0.png'
path = os.path.join(path, name)

# Using cv2.imread() method to the path 
# but we only want de green 'g' channel
# the others are 0
img = cv2.imread(path, -1)
print('image shape og:', img.shape)
img = img[74:580,78:585,1]

print('img:', img.shape)
print('len shape', len(img.shape))
print('hello ')

plt.imshow(img)
plt.show()
