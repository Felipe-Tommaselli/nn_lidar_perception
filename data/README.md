# Data

The data sources for the Neural Network are located in this directory. It is important to note that we utilized the `lidar2images.py`, `create_dataset.py` and the `lidar_tag.py` scripts to transform the raw data to this format. 
In general, we have two formats in this directory:
* `tags`: The "Label_DataN.csv" where the labels of each image are stored. This Labels were generated manually or artifically depending on the dataset.
* `images`: The "trainN" folder where the images are stored. The images are stored in the following format: "trainN/imageM.png" where N is the number of the dataset and M each image.

It's worth notion that the artificial data are generated with fixed point distribuition. 

## Dataset Description

### Recorded Data

Each dataset has a utility for the neural network pipeline, this dataset are all real data recorded with RosBags from field test and manually labeled. The datasets are described in the following table:

| Data     | Labeled? | N. Labels | Description |
|----------|----------|-----------|-------------|
| train1   |   Yes    |   1729    |   Original Dataset    |
| train2   |   Yes    |   1611    |   Puerto Rico dataset    |
| train3   |   No     |   -       |   More of Puerto Rico    |
| train4   |   No     |   -       |   More of Puerto Rico    |
| train5   |   No     |   -       |   More of Puerto Rico    |


### Artificial Data 

The artificial data are generated with the `artificial_generator.py` script. This script generates a dataset with a fixed number of images and labels. The labels are generated with a fixed point distribution. The datasets are described in the following table:

| Data     | N. Labels | Description |
|----------|-----------|-------------|
| train    |  27000    |   Original code with the old labeling, prettly solid but we got best.    |
| train2   |  45500    |   Hard training setup, near to 50k images.    |
| train3   |  245      |   Small training setup to fast tests.    |
| train4   |  19500    |   Medium setup that satisfy most of the tests.    |

*obs: All the datasets are labeled automatically with the script*