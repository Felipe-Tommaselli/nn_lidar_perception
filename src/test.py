import cv2
import os
import matplotlib.pyplot as plt

SLASH = '/'

# open the image wiith cv2 and show it
if os.getcwd().split(SLASH)[-1] == 'src':
    os.chdir('..')
path = ''. join([os.getcwd(), SLASH, 'assets', SLASH, 'images', SLASH])

name = 'image'+str(10)+'.png'
path = os.path.join(path, name)

print('path: ', path)

# Using cv2.imread() method to the path 
# but we only want de green 'g' channel
# the others are 0
img = cv2.imread(path, -1)
print('image shape og:', img.shape)
img = img[74:580,78:585,1]

print('img:', img.shape)
print('len shape', len(img.shape))

plt.imshow(img)
plt.show()