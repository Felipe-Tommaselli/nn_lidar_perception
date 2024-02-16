# IC_NN_LIDAR: Analytics

> Package developed to provide insights on different aspects of the project through data analytics.

## Description

This directory contains Jupyter Notebook (.ipynb) files, that basically allow the code to be interactive and the results to be visualized.

## Projects

- `model_test.ipynb`: Test the performance of different models architectures. The used metrics were "Number of parameters" for "Mean error in validation".The evalueted models were:
    - Mobile Net v2
    - EfficientNet B0 3
    - ResNet18
    - VggNet16
- `parametrization.ipynb`: This notebook will analyze the parametrization of the model. For that we will simulate a range of line equations parametrization to determine the best one. With this, we always garantee that the worst guest of the neural network will be at least in the range of the data.
- `boxes_corridor.ipynb`: Notebook to test specific images generated during the simulator test. Besides the inference test, this notebook was also used to test the perspective of the image for example.
- `data.ipynb`: This notebook for simple it looks, represents a big innovation on the project. With the `data.csv` data from real life test, I noticed that the inversion of the line equation, changed significantly the data distribution. I changed Y(x) to X(y) and you can see the test by the "range" of the variables. 

## Summary

To summarize, the analytics package don't contain any active projects, but contains old implementations that achived different goals.

