# IC_NN_Lidar

Population growth disproportionate to the increase in agricultural production requires the modernization of the technologies used. In this field, terrestrial robotics stands out for its versatility of applications to increase the sector's productivity. Featuring versatility on a single platform, the TerraSentia robot is a solution for navigation under canopy (under the canopy of plantations), an environment in which the lack of trust in sensors and the irregularity of the scene make navigation and navigation difficult. robot control. Thus, the use of a 2D LiDAR (Light Detection and Ranging) emerges as a resource to overcome the adversities described, in addition to enabling data collection in low-light environments.

That said, this Scientific Initiation project proposes the use of a Convolutional Neural Network on data from a 2D LiDAR sensor for navigation on the TerraSentia platform. Therefore, an increase in TerraSentia's performance in under canopy experiments is expected, compared to the state of the art through algorithms using only geometric heuristics instead of the Neural Network.

The initial approach identifies a perception system, in which the robot obtains information from the environment only through LiDAR, seeking alternatives for navigation without GNSS and without the use of a camera. In general, this approach, when achieving viable results, must be integrated into the TerraSentia platform, that is, the perception data must be loaded into a Path Planning algorithm and then into a Controller.

With the implemented approach, simulation results (using the Gazebo software) can generate system quality analyses. Furthermore, real tests with TerraSentia itself, as already carried out in the state of the art, will be able to measure the proposed alternative. With this, it is expected to expand the academic state of the art of land navigation, proving and comparing the viability of all technologies used.

> This project was supported by grants no. 2022/08330-9, São Paulo Research Foundation (FAPESP). The authors express his acknowledgments to EarthSense and Mobile Robotics Laboratory (LabRoM) from Engineering School of São Carlos (EESC-USP).

## Installation

### Dependencies

``` shell
sudo apt install python3.8
sudo apt install python3-pip
```

```
pip install --upgrade setuptools
```

``` shell
pip install customtkinter==5.1.2 matplotlib==3.7.1 numpy==1.24.2 pandas==2.0.0 Pillow==9.5.0 scikit_learn==1.2.2 scipy==1.10.1 torch==2.0.0 torchsummary==1.5.1 torchvision==0.15.1
```

``` shell
pip install opencv-contrib-python>=4.7.0
```
