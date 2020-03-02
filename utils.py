import os
import glob
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np

EPOCHS = 50
BS = 8
SIZE = 300 ## Resize factor
TEST_SIZE = 0.2
label_size=4

video_paths=glob.glob("train_videos/*/*.mp4")
for i,video_fp in tqdm(enumerate(video_paths)):
    folder_path="train/"+video_fp.split("/")[-2]+"_train"
    if folder_path not in os.listdir():
        os.mkdir(folder_path)
    vidcap = cv2.VideoCapture(video_fp)
    success,image = vidcap.read()
    count = 0
    while success:
        if count%10==0:
            cv2.imwrite(folder_path+"/"+str(i)+"frame%d.jpg" % count, image)
        success,image = vidcap.read()
        count += 1

img_paths=glob.glob("train/*_train/*")
labels=[i.split("/")[-2].split("_")[0] for i in img_paths]
train_csv=pd.DataFrame()
train_csv["ImageName"]=img_paths
train_csv["labels"]=labels
#train_csv["count"]=[1]*len(train_csv)
#train_csv=pd.pivot_table(train_csv,index="ImageName",values="count",columns="labels").fillna(0).astype(int).reset_index()
train_csv.to_csv("train_csv.csv",index=False)
