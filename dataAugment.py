from skimage import transform
import numpy as np
from matplotlib import pyplot as plt
import glob

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img_list = glob.glob("test_images/*.jpg")
images = [rgb2gray(plt.imread(ele)) for ele in img_list]
print(len(images))

shift_y, shift_x = (np.array(images[0].shape)-1) / 2.   #Assumption: All images of same size
tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(60))
tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])

for image in images:
    image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, order = 3)

    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(image_rotated)
    plt.show()

    print("original image maximum at: ", np.unravel_index(np.argmax(image), image.shape))
    print("rotated image maximum at : ", np.unravel_index(np.argmax(image_rotated), image_rotated.shape))