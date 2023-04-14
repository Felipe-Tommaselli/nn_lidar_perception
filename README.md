# IC_NN_Lidar

Population growth disproportionate to the increase in agricultural production requires the modernization of technologies used in agriculture. In this branch, terrestrial robotics stands out for its versatility of applications to increase the sector's productivity. Featuring versatility and accessibility on a single platform, the TerraSentia robot is a solution for under canopy navigation (under the canopy of plantations), an environment in which the lack of confidence in sensors and the irregularity of the scenery make it difficult to navigate and control the robot. So, the
The use of a 2D LiDAR (Light Detection and Ranging) emerges as a resource to overcome the adversities described, in addition to enabling data collection in low-light environments.

That said, this Scientific Initiation project proposes the use of a Convolutional Neural Network in data from a 2D LiDAR sensor for navigation on the TerraSentia platform. Thus, it is proposed the creation of 2 systems (Pre Neural Network data processing and the Neural Network itself). Therefore, TerraSentia performance is expected to increase in under canopy experiments, compared to the state of the art through algorithms using only the heuristic
instead of the Neural Network.

## Installation

### Dependencies

``` shell
sudo apt install python3.8
sudo apt install python3-pip
```

``` shell
pip install customtkinter==5.1.2 matplotlib==3.7.1 numpy==1.24.2 pandas==2.0.0 Pillow==9.5.0 scikit_learn==1.2.2 scipy==1.10.1 torch==2.0.0 torchsummary==1.5.1 torchvision==0.15.1
```

``` shell
pip install opencv-contrib-python>=4.7.0
```
