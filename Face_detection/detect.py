import os
import numpy as np
import tensorflow as tf
import cv2

from Face_detection.cnn import Test
from Face_detection.Load_data import Train_Test_sets


model_path = "model/"

Train_set, Test_set = Train_Test_sets()

X = []
y = []
for data in Train_set:
    X.append(data[0])
    y.append(data[1])
    # print(data[1])

X = np.array(X)
y = np.array(y)
X = X.reshape(-1, 32, 32, 1)
print(X.shape)
Test(X,y,model_path)