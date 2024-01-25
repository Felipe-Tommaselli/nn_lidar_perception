#!/usr/bin/env python3

import os
import time
import cv2
import torch
import numpy as np
import math 
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns 
import torchvision.models as models

from sensor_msgs.msg import LaserScan

import rospy

device = torch.device("cpu")

os.chdir('..')
print(os.getcwd())

global fid
fid = 5

############### ROS INTEGRATION ###############

############### MODEL LOAD ############### 

def load_model():
    ########### MOBILE NET ########### 
    model = models.mobilenet_v2()
    model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # MobileNetV2 uses a different attribute for the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(512, 256),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(256, 3)
    )

    path = os.getcwd() + '/models/' + 'model_005_17-01-2024_15-38-12.pth'
    checkpoint = torch.load(path, map_location='cpu')  # Load to CPU
    model.load_state_dict(checkpoint)
    model.eval()

    return model

############### DATA EXTRACTION ###############

def generate_image(data):

    lidar = data.ranges
    
    min_angle = np.deg2rad(0)
    max_angle = np.deg2rad(180) # lidar range
    angle = np.linspace(min_angle, max_angle, len(data), endpoint = False)

    # convert polar to cartesian:
    # x = r * cos(theta)
    # y = r * sin(theta)
    # where r is the distance from the lidar (x in lidar)
    # and angle is the step between the angles measure in each distance (angle(lidar.index(x))
    x_lidar = [x*np.cos(angle[lidar.index(x)]) for x in lidar]
    y_lidar = [y*np.sin(angle[lidar.index(y)]) for y in lidar]

    POINT_WIDTH = 18
    if len(xl) > 0:
        plt.cla()
        plt.plot(xl,yl, '.', markersize=POINT_WIDTH, color='black')
        plt.axis('off')
        plt.xlim([-1.5, 1.5])
        plt.ylim([0, 2.2])
        plt.grid(False)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.gcf().set_size_inches(5.07, 5.07)
        plt.gcf().canvas.draw()

        plt.savefig('temp_image')


def get_image():
    image = cv2.imread("temp_image.png")

    os.remove("temp_image.png")

    # convert image to numpy 
    image = np.array(image)

    # crop image to 224x224 in the pivot point (112 to each side)
    # image = image[100:400, :, :]
    image = image[:,:, 1]
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # add one more layer to image: [1, 1, 224, 224] as batch size
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    # convert to torch
    image = torch.from_numpy(image).float()
    return image

############### INFERENCE AND PLOT ###############

def deprocess(image, label):
    ''' Returns the deprocessed image and label. '''

    if len(label) == 3:
        # we suppose m1 = m2, so we can use the same deprocess
        #print('supposing m1 = m2')   
        w1, q1, q2 = label
        w2 = w1
    elif len(label) == 4:
        #print('not supposing m1 = m2')        
        w1, w2, q1, q2 = label

    # DEPROCESS THE LABEL
    q1_original = ((q1 + 1) * (187.15 - (-56.06)) / 2) + (-56.06)
    q2_original = ((q2 + 1) * (299.99 - 36.81) / 2) + 36.81
    w1_original = ((w1 + 1) * (0.58 - (-0.58)) / 2) + (-0.58)
    w2_original = ((w2 + 1) * (0.58 - (-0.58)) / 2) + (-0.58)

    #print(f'labels w1={w1}, w2={w2}, q1={q1}, q2={q2}')
    m1 = 1/w1_original
    m2 = 1/w2_original
    b1 = -q1_original / w1_original
    b2 = -q2_original / w2_original

    label = [m1, m2, b1, b2]

    return label

def inference(image, model):
    # Inicie a contagem de tempo antes da inferência
    start_time = time.time()

    # get the model predictions
    predictions = model(image)

    # Encerre a contagem de tempo após a inferência
    end_time = time.time()

    #print('Inference time: {:.4f} ms'.format((end_time - start_time)*1000))

    return predictions

def prepare_plot(x, predictions, image):
    # convert the predictions to numpy array
    predictions = predictions.to('cpu').cpu().detach().numpy()
    predictions = deprocess(image=image, label=predictions[0].tolist())


    # convert image to cpu 
    image = image.to('cpu').cpu().detach().numpy()
    # image it is shape (1, 1, 507, 507), we need to remove the first dimension
    image = image[0][0]

    # line equations explicitly

    # get the slopes and intercepts
    m1p, m2p, b1p, b2p = predictions

    # get the x and y coordinates of the lines
    y1p = m1p*x + b1p
    y2p = m2p*x + b2p

    return y1p, y2p, image

def show(x, y1p, y2p, image):
    linewidth = 2.5

    ax.plot(x, y1p, color='red', label='Predicted', linewidth=linewidth)
    ax.plot(x, y2p, color='red', linewidth=linewidth)

    border_style = dict(facecolor='none', edgecolor='black', linewidth=2)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, **border_style, transform=ax.transAxes))
    plt.legend(loc='upper right', prop={'size': 9, 'family': 'Ubuntu'})
    ax.imshow(image, cmap='magma', norm=PowerNorm(gamma=16), alpha=0.65)
    ax.axis('off')
    plt.show()


def lidar_callback(data):

    model = load_model()
    
    generate_image(data)
    image = get_image()
    predictions = inference(image, model)
    y1p, y2p, image = prepare_plot(x, predictions, image)

    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(y1p)
    line2.set_xdata(x)
    line2.set_ydata(y2p)

    ax.imshow(image, cmap='magma', norm=PowerNorm(gamma=16), alpha=0.65)

    plt.title(f"Inference {int(t//2)}/{file_count}", fontsize=22)

    # drawing updated values
    fig.canvas.draw()

    fig.canvas.flush_events()
    time.sleep(0.05)

############### MAIN ###############

if __name__ == '__main__':

    ########## PLOT ########## 
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 5), frameon=True)
    x = np.arange(0, 224)
    linewidth = 2.5

    # create the lines with rand values
    line1, = ax.plot(x, x, color='red', label='Predicted', linewidth=linewidth)
    line2, = ax.plot(x, x, color='red', linewidth=linewidth)
    image = np.zeros((224, 224)) # empty blank (224, 224) image

    border_style = dict(facecolor='none', edgecolor='black', linewidth=2)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, **border_style, transform=ax.transAxes))
    plt.legend(loc='upper right', prop={'size': 9, 'family': 'Ubuntu'})
    ax.axis('off')
    ax.imshow(image, cmap='magma', norm=PowerNorm(gamma=16), alpha=0.65)

    rospy.init_node('RTinference', anonymous=True)
    rospy.Subscriber('/terrasentia/scan', LaserScan, lidar_callback)
    rospy.spin()
