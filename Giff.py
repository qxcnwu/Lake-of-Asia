import numpy as np
import os,shutil
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from images2gif import writeGif
import imageio
import cv2

def tif_to_jpg(tif_path,jpg_path):
    im=Image.open(tif_path)
    im.save(jpg_path)
    print('success turn file')

def add_word(jpg_path,year):
    font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc",200)
    im=Image.open(jpg_path)
    draw=ImageDraw.Draw(im)
    draw.text((2000,0),str(year),(255,255,255),font=font)
    draw = ImageDraw.Draw(im)
    im.save(jpg_path)

gif_parh='C:/Users/邱星晨/Desktop/pic/1.gif'
jpg_name=[]
year_list=['巴尔喀什湖1985','巴尔喀什湖1990','巴尔喀什湖1995','巴尔喀什湖2000','巴尔喀什湖2005','巴尔喀什湖2010','巴尔喀什湖2015']
for i in range(7):
    tif_path = 'C:/Users/邱星晨/Desktop/pic/' + str(i + 1) + '.tif'
    jpg_path = 'C:/Users/邱星晨/Desktop/pic/' + str(i + 1) + '.jpg'
    tif_to_jpg(tif_path, jpg_path)
    add_word(jpg_path,year_list[i])
    jpg_name.append(jpg_path)

frames=[]
for name in jpg_name:
    frames.append(imageio.imread(name))
imageio.mimsave(gif_parh,frames,fps=1)