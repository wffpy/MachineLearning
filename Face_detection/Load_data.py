import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

face_path = "dataset/Face/faces/"
scence_path = "dataset/Negetive"

GOOD = [1,0]
BAD = [0,1]
SNAP_COUNT = 5
MIN_LEN = 10
IN_SIZE = (32,32)
Train_NUM = 10000
Train_split = int(0.6*Train_NUM)


def rand_snap(img):
    r = []
    x = img.shape[0]
    y = img.shape[1]
    #Generate 5 snapshots of different sizes
    for i in range(SNAP_COUNT):
        snap_size = max([MIN_LEN,int(random.random()*200)])
        fx = int(random.random()*(x-snap_size))
        fy = int(random.random()*(y-snap_size))
        snap = img[fx:fx+snap_size,fy:fy+snap_size]
        r.append(cv2.resize(snap,IN_SIZE))
    return r


def Load_face_image(face_path):
    faces = []
    for root,dirs,files in os.walk(face_path):
        print(root)
        for file in files:
            img = cv2.imread(os.path.join(root,file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_new = cv2.resize(gray,(32,32),interpolation=cv2.INTER_CUBIC)
            faces.append((img_new,GOOD))
            # rand_snap(gray)
        random.shuffle(faces)
    faces = faces[:Train_NUM]
    return faces



def Load_sence_images(root_path):
    scences = []
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            img_dir = os.path.join(root,dir)
            for root2, dirs2, files2 in os.walk(img_dir):
                for file2 in files2:
                    img = cv2.imread(os.path.join(root2,file2))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_new = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_CUBIC)
                    scences.append((img_new,BAD))


                    for r in rand_snap(gray):
                        scences.append((r,BAD))
    random.shuffle(scences)
    scences = scences[:Train_NUM]
    # print(scences[1][1])

    return scences


def Train_Test_sets():
    faces = Load_face_image(face_path)
    scences = Load_sence_images(scence_path)


    TrainX = faces[:Train_split]
    TrainX.extend(scences[:Train_split])
    random.shuffle(TrainX)


    TestX = faces[Train_split:]
    TestX.extend(scences[Train_split:])
    random.shuffle(TestX)

    # print(TrainX)


    return TrainX,TestX

# Train_Test_sets()