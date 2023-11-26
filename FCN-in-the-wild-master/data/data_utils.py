
# the color map and class map refers to
# https://bitbucket.org/visinf/projects-2016-playing-for-data/overview
# the matlab code could be found in 'projects-2016-playing-for-data\label\initLabels.m'
#

import numpy as np
from PIL import Image as image


#-------------------------------
# B    G    R      label
#-------------------------------
# 64   128  64     Animal
# 192  0    128    Archway
# 0    128  192    Bicyclist
# 0    128  64     Bridge
# 128  0    0      Building
# 64   0    128    Car
# 64   0    192    CartLuggagePram
# 192  128  64     Child
# 192  192  128    Column_Pole
# 64   64   128    Fence
# 128  0    192    LaneMkgsDriv
# 192  0    64     LaneMkgsNonDriv
# 128  128  64     Misc_Text
# 192  0    192    MotorcycleScooter
# 128  64   64     OtherMoving
# 64   192  128    ParkingBlock
# 64   64   0      Pedestrian
# 128  64   128    Road
# 128  128  192    RoadShoulder
# 0    0    192    Sidewalk
# 192  128  128    SignSymbol
# 128  128  128    Sky
# 64   128  192    SUVPickupTruck
# 0    0    64     TrafficCone
# 0    64   64     TrafficLight
# 192  64   128    Train
# 128  128  0      Tree
# 192  128  192    Truck_Bus
# 64   0    64     Tunnel
# 192  192  0      VegetationMisc
# 0    0    0      Void
# 64   192  0      Wall
#---------------------------------
classes = [
    'unlabeled','ego vehicle','rectification border','out of roi','static',
    'dynamic','ground','road' ,'sidewalk','parking',
    'rail track','building', 'wall','fence' ,'guard rail',
    'bridge','tunnel','pole','polegroup','traffic light',
    'traffic sign', 'vegetation','terrain','sky' ,'person',
    'rider', 'car','truck','bus' ,'caravan', 
    'trailer','train', 'motorcycle' 'bicycle','license plate']

colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],           \
          [20, 20, 20], [111, 74, 0], [81, 0, 81],              \
          [128, 64, 128], [244, 35, 232], [250, 170, 160],      \
          [230, 150, 140], [70, 70, 70], [102, 102, 156],       \
          [190, 153, 153], [180, 165, 180], [150, 100, 100],    \
          [150, 120, 90], [153, 153, 153], [153, 153, 153],     \
          [250, 170, 30], [220, 220, 0], [107, 142, 35],        \
          [152, 251, 152], [70, 130, 180], [220, 20, 60],       \
          [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],   \
          [0, 0, 90], [0, 0, 110], [0, 80, 100], [0, 0, 230],   \
          [119, 11, 32], [0, 0, 142]]
labelClasses = [5,6,7,8,9,12,13,14,15,16,17,18,20,21,22,23,24,  \
                25,26,27,28,29,31,32,33,34,35]


def index2rgb(indexed, palette=colors):
  w, h = indexed.shape
  rgb_img = np.zeros((w, h, 3))

  for i in range(len(palette)):
    mask = indexed == i
    rgb_img[mask] = palette[i]

  return rgb_img

def get_lable(i):
  return classes[labelClasses[i]]

def get_label_classes():
  return labelClasses

def resize_input(img):
  return img.resize((1000,1000),resample=image.BILINEAR)

def resize_output(img, size=(1052, 1914)):
  # pdb.set_trace()
  if isinstance(img, torch.FloatTensor):
    np_array = img.numpy().astype('uint8')
    np_array = np_array.transpose((1,2,0))
    img = image.fromarray(np_array,mode='RGB')
  elif isinstance(img, np.ndarray):
    # print(img.shape)
    img = image.fromarray(img,mode='RGB')
  # print(type(img))
  return img.resize(size,resample=image.BILINEAR)

if __name__ == '__main__':
  print(len(colors), len(labelClasses))