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

device = torch.device("cpu")

os.chdir('..')
print(os.getcwd())

global fid
fid = 5

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
    torch.nn.Linear(512, 3)
    )

    path = os.getcwd() + '/models/' + 'model_005_26-01-2024_21-37-22.pth'
    checkpoint = torch.load(path, map_location='cpu')  # Load to CPU
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def get_data(t:int, path: str):
    image = cv2.imread(os.path.join(path, f"image{t}.png"))

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

if __name__ == '__main__':

    model = load_model()

    # Count the number of files in the folder
    path = os.path.join(os.getcwd(), "data", "gazebo_data", f"train{fid}")
    file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    print(f"Number of files in the folder: {file_count}")

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

    for t in range(2, file_count, 20):
        image = get_data(t, path)
        predictions = inference(image, model)
        y1p, y2p, image = prepare_plot(x, predictions, image)

        # updating data values
        line1.set_xdata(x)
        line1.set_ydata(y1p)
        line2.set_xdata(x)
        line2.set_ydata(y2p)

        #print("y1p:", y1p[0])
        #print("y2p:", y2p[0])

        ax.imshow(image, cmap='magma', norm=PowerNorm(gamma=16), alpha=0.65)

        plt.title(f"Inference {int(t//2)}/{file_count}", fontsize=22)

        # drawing updated values
        fig.canvas.draw()

        fig.canvas.flush_events()
        time.sleep(0.05)

