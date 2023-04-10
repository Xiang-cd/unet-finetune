import os
import re
import time
import random
import threading
import imghdr
import clip
import torch
import argparse, os, sys, glob
import clip
from tqdm import tqdm
from PIL import Image
import numpy as np


            
            
class DreamWorker(threading.Thread):
    def __init__(self, project_base, gpu_index, data_root):
        super(DreamWorker, self).__init__()
        self.device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = clip_model
        self.preprocess = preprocess
        self.project_base = project_base
        self.data_root = data_root
        self.res = []
    
    def image_similarity(self, img1, img2):
        with torch.no_grad():
            image_f1 = self.model.encode_image(img1).type(torch.float32)
            image_f2 = self.model.encode_image(img2).type(torch.float32)
        normed1 = image_f1 / torch.norm(image_f1)
        normed2 = image_f2 / torch.norm(image_f2)
        return torch.sum(normed1 * normed2)

    def mean_similarity(self, dir1_name, dir2_name):
        dir1 = os.listdir(dir1_name)
        dir2 = os.listdir(dir2_name)
        # print(dir1, dir2, sep="\n")
        num = 0
        total_sim = 0
        image1_list = []
        image2_list = []
        for file1 in dir1:
            file1 = os.path.join(dir1_name, file1)
            if os.path.isfile(file1) and imghdr.what(file1) in ["jpeg", "png", "jpg"]:
                image1_list.append(Image.open(file1))
        for file2 in dir2:
            file2 = os.path.join(dir2_name, file2)
            if os.path.isfile(file2) and imghdr.what(file2) in ["jpeg", "png", "jpg"]:
                image2_list.append(Image.open(file2))
        print(f"img1 num:{len(image1_list)}, img2 num:{len(image2_list)}")
        
        for file1 in image1_list:
            image1 = self.preprocess(file1).unsqueeze(0).to(self.device)
            for file2 in image2_list:
                image2 = self.preprocess(file2).unsqueeze(0).to(self.device)
                num = num + 1
                total_sim += self.image_similarity(image1, image2)
        return total_sim / num
    def run(self):
        image_base = os.path.join(self.project_base, "images")
        if not os.path.exists(image_base):
            print("no image path in project", self.project_base)
        def to_number(s):
            return int(s)
        dirs = os.listdir(image_base)
        dirs.sort(key=to_number)
        print("listed image dirs", dirs)
        for step in dirs:
            img_dir = os.path.join(self.project_base, "images", str(step))
            if not os.path.exists(img_dir):
                print(f"{img_dir} not existes!")
                continue
            mean_sim = self.mean_similarity(img_dir, self.data_root)
            self.res.append(mean_sim.cpu().numpy())