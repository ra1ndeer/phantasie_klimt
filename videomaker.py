import cv2 
import numpy as np
from natsort import natsorted
from PIL import Image
from os import listdir
from os.path import isfile, join



mypath = "./output/images"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


imgs_for_video = list()
for f in natsorted(onlyfiles):
    imgs_for_video.append(Image.open(f"output/images/{f}").resize((1024, 1024), Image.ANTIALIAS))


imgs_for_video = [cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR) for i in imgs_for_video]

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter("output/sampling_progress.mp4", fourcc, 60, (1024, 1024))
for i in range(len(imgs_for_video)):
    print(f"Adding image {i} of {len(imgs_for_video)}...")
    out.write(imgs_for_video[i])
out.release()