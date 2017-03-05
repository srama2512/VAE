import cv2
import numpy 


#The following function does the image preprocessing to convert from 210x160x3 to 84x84
# Note: This is the preprocessing that was used to train the expert network from Breakout
# The frames sampled on DQN expert play for training the VAE underwent this preprocessing.
# Thus, to test Thompson Sampling/Linear Q Learning on top of this representation, it is important to 
# use the same preprocessing as what was used to train the beta-VAE.

def preprocessing(image, resize_width = 84, resize_height = 84):
    # input: image of size 210 x 160 x 3
    # resize dimensions: 84 x84
    # We use cv2.resize function with linear interpolation
    resized_image = cv2.resize(image, (resize_width, resize_height), interpolation = cv2.INTER_LINEAR)
    return resized_image


def process_42(image):
    # input: image of size 210x160
    # works for both color and grayscale: dimensionality: shape_1 x shape_2 x n_c (=1 for gray,3 color)
    # resize dimensions: 42x42, convert to grayscale if color

    shape = images.shape

    image = image[34:34+160, :160]
    image = cv2.resize(image, (80,80))
    image = cv2.resize(image, (42,42))
        
    image = image.mean(2) #taking mean across the 3rd (channel) dimension. 

    # dimension of image now is 42x42, need to convert it back to 42x42x1

    image = np.reshape(image, [42,42,1])
    return image

