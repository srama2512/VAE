from skimage import transform
import numpy as np
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

image = rgb2gray(plt.imread("lena.bmp"))
print(image.shape)

shift_y, shift_x = (np.array(image.shape)-1) / 2.
tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(60))
tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, order = 3)


plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(image_rotated)
plt.show()

print("original image maximum at: ", np.unravel_index(np.argmax(image), image.shape))
print("rotated image maximum at : ", np.unravel_index(np.argmax(image_rotated), image_rotated.shape))