import os
from glob import glob 
import shutil 
from tqdm import tqdm 

# Train
# path = "../data/tiny-imagenet-200/"

# train = glob(path + "train/*")

# train.sort()

# for directory in tqdm(train, total = len(train)):
#     name = directory.split("/")[-1]
#     if name != "n01443537" and name != "n01629819":
#         os.remove(directory + "/" + name + "_boxes.txt")
#         for im in glob(directory + "/images/*"):
#             shutil.move(im, "../data/tiny-imagenet-200/train/" + name + "/" + im.split("/")[-1])
#         os.rmdir(directory + "/images")

# Test 

# path = "../data/tiny-imagenet-200/"

# text_file = open(path + "val/val_annotations.txt", "r")
# lines = text_file.readlines()

# names = dict()

# for l in lines: 
#     jpg = l.split("\t")[0]
#     folder = l.split("\t")[1]
#     names[jpg] = folder

# valid = glob(path + "val/images/*.JPEG")
# valid.sort()

# for img in tqdm(valid, total = len(valid)):
#     name = img.split("/")[-1]
#     f = names[name]
#     if not os.path.exists(path + "val/" + f):
#         os.mkdir(path + "val/" + f)
#     shutil.move(img, "../data/tiny-imagenet-200/val/" + f + "/" + name)

