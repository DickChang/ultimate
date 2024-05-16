import cv2 as cv
from skimage.io import imread, imshow, imsave
from skimage import transform
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
import json

root_dir = os.path.normpath("E:\\ultimate\\ultimate")
data_dir = os.path.join(root_dir, "data")

##################
# Read in the data json
##################
data_f = open(os.path.join(data_dir, 'data.json'))
data = json.load(data_f)
data_f.close()

##################
# Read in the src image and coords
##################
#src_img_name = "20211211_SEAatSD_0000001451.jpg"
#src_img_name = "20240330_COatSD_14h15m20s084.png"
#src_img_name = "20240330_COatSD_calib1.png"
#src_img_name = "20240330_COatSD_calib2.png"
#src_img_name = "20240330_COatSD_calib3.png"
#src_img_name = "20240503_UFA_SLCatSD_011_calib1.png"
#src_img_name = "20240503_UFA_SLCatSD_011_calib2.png"
src_img_name = "20240503_UFA_SLCatSD_011_calib3.png"
found = False
for example in data['examples']:
    if example["img_name"] == src_img_name:
        found = True
        break
if found == False:
    print("Could not find src_img_name in data.json\nExiting")
    exit()

src_img = imread(os.path.join(data_dir, src_img_name))
if src_img is None:
    sys.exit("Could not read the image.")
plt.figure(figsize=(17,7))
imshow(src_img, origin='upper')
src = np.array(example["src_nparray"])
plt.plot(src[:, 0], src[:, 1], 'ro')
plt.xlim(0, src_img.shape[1])
plt.ylim(src_img.shape[0], 0)
plt.show() # show the original image with red dots to show src_nparray

##################
# Read in the dst image and coords
##################
field = imread(os.path.join(root_dir,example["dst_field_img_name"]))
if field is None:
    sys.exit("Could not read the image.")
plt.figure(figsize=(17,7))
imshow(field, aspect='auto')
dst = np.array(example["dst_nparray"])
plt.plot(dst[:, 0], dst[:, 1], 'ro')
plt.xlim(0, field.shape[1])
plt.ylim(field.shape[0], 0)
plt.axis('scaled')
plt.show() # show the dst_field_img_name with red dots to shot dst_nparray


##################
# Calculate the transform
##################
tform = transform.estimate_transform('projective', src, dst)
tf_img = transform.warp(src_img, tform.inverse, output_shape=(field.shape))
fig, ax = plt.subplots()
ax.imshow(tf_img)
_ = ax.set_title('projective transformation')
plt.savefig(os.path.join(root_dir,"ultimate_test.png"),bbox_inches='tight',dpi=250)
plt.show() # show the original image transformed and projected on top of the field dst_img


##################
# Use the transform to find positions of players
##################
player_coords_src = np.array(example["player_coords_nparray"])
player_coords_dest = np.array([tform(coord)[0] for coord in player_coords_src]) # do the transform
plt.figure(figsize=(17,7))
imshow(field, aspect='auto')
plt.plot(player_coords_dest[:, 0], player_coords_dest[:, 1], 'ro')
plt.xlim(0, field.shape[1])
plt.ylim(field.shape[0], 0)
plt.axis('scaled')
plt.savefig(os.path.join(root_dir,"ultimate_test_2.png"),bbox_inches='tight',dpi=250)
plt.show() # show the final placement of players on the dst image

#k = cv.waitKey(0)
#if k == ord("s"):
    #cv.imwrite("ultimate_test.png", ultimate)
    #imsave("ultimate_test.png", ultimate)
