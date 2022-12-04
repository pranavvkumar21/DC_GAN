#!/usr/bin/env python3


import glob
from PIL import Image
import os

img_folder = "./test/test/"

def make_gif(frame_folder):
    filenames = os.listdir(frame_folder)
    filenames.sort()
    #print(filenames)
    os.chdir(frame_folder)
    frames = [Image.open(image) for image in filenames]
    frame_one = frames[0]
    frame_one.save("../my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == "__main__":
    make_gif( img_folder)
