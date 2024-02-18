# IC_NN_LIDAR: Src (Source)

> This package contains the "source/script" of the project. The files in this folder are basically about the **training** of the Neural Network models. 

## Folder archtecture

This folder is as follows:
* parent: `/src`
* child 1: `/utils`
* child 2: `/test`

With that, all files in the `src` folder, are related to the main files in the training pipeline:

### Training `/src` scripts

We have 5 scripts: `main.py`, `dataloader.py`, `pre_process.py` (training) and `nn_test.ipynb`, `test_dataloader.py` (eval).
The `main.py` script runs the neural network params and the model_fit (the heart of the project). This script also uses the 
`dataloader.py` class to create the dataset. Finally, the `dataloader.py` uses some functions on the `pre_process.py`, but it is worth noticing that the `pre_process.py` contains a library of functions for different purposes.

On the other hand, the `nn_test.ipynb` and `test_dataloader.py` are used to test the model performance with a specific image. As we only need one load of the dataset, we can't afford the `dataloader.py` to carry all images into RAM memory. For that, the `test_dataloader.py` does exactly the same thing as his brother, but for one single image. 

### `/utils` scripts

This folder contains different scripts for many applications united by their usefulness. In general, the `artificial_generator.py`, `artificial_test.ipynb`, `create_dataset.py`, `lidar_tag.py` and `lidar2images.py` do not share much in common, but they are handy scripts that are used often (not every time). For that, they are classified as "utils".

* `artificial_generator.py`: Create the dataset based on some parameters in the script that generate a new raw_data dataset full of images and labels;
* `artificial_test.ipynb`: Test the points distributions before using the `artificial_generator.py`;
* `create_dataset.py`: Create a real-time dataset based on the `/terrasentia/scan` rostopic;
* `lidar_tag.py`: Generate the label in the .csv format by hand-made labelling of a specific image (from raw_data). Used in real-life dataset training;
* `lidar2images.py`: Provide specific resources to the `lidar_tag.py` file. 

It is worth noticing that both `lidar_tag.py` and `lidar2images.py` are deprecated and have not been used for a long time since there is no need to use hand-made labelling anymore.

### `/ŧest` scripts

The test scripts folder contains active projects that were not deprecated but can't go directly to running. These side projects vary from time to time and typically are not well-documented. However, once they complete their purpose and go to deploy, they are moved to another folder, and the documentation starts to get serious. 
