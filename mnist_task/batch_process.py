import os
import numpy as np
import time

beta_vals = np.hstack(([0.1,0.5,0.8],np.arange(1,11,1)))

for ele in beta_vals:
    start_time = time.time()
    os.system("python train_augmented_mnist.py --vae_model=cnn --beta={:}".format(ele))
    os.system("python testAugData.py --vae_model=cnn --beta={:}".format(ele))
    print('\n\n\n\n\n\n\n\n\n\nBeta={:} is done. Time taken: {:}'.format(ele, time.time()-start_time))
