#import scipy.misc
from matplotlib.pyplot import imread, imsave
import os,sys,glob,shutil
import numpy as np
from PIL import Image
from skimage.transform import resize
import scipy.misc

def main():      
   images_path = '../../../data/CelebA/img_align_celeba_png/*.png'
   images = glob.glob(images_path)
   images = sorted(images)
   num = len(images)
   print(num)

   dir_path = '../../../data/CelebA/centered_celeba_64_all'

   if os.path.exists(dir_path):
      shutil.rmtree(dir_path)
   os.makedirs(dir_path)
   patch_size = 64 #64

   for i in range(num):
       img = imread(images[i])
       h,w = img.shape[:2]
       if h>w:
          j = (h-w)//2
          temp = resize(img[j:-j,:,:],[patch_size,patch_size])
       else:
          i = (w-h)//2
          temp = resize(img[:,j:-j,:],[patch_size,patch_size])
       _, name = os.path.split(images[i])
       name = name.split('.')[0]       
       name = name + '.png'
       save_name = os.path.join(dir_path, name)
       scipy.misc.imsave(save_name, temp)


       if i%2000==0:
        print('%d/%d... Processing image:%s.'%(i,num,save_name))



if __name__ == "__main__":
    main()
