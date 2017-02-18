import argparse
import matplotlib.pyplot as plt
import numpy as np

z_mean_theta = np.load("z_mean_theta.npy")
mean_of_std_devs = np.load("mean_of_std_devs.npy")
z_var = np.load("z_var.npy")


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='generated/iter_100000/')
parser.add_argument('--input_h5', default='data/augmented_mnist.h5')
parser.add_argument('--vae_model', default='mlp', help='[ mlp | cnn ]')
parser.add_argument('--num_rotations', default=20, type=int)
parser.add_argument('--num_translation_x', default=15, type=int)
parser.add_argument('--num_translation_y', default=15, type=int)
parser.add_argument('--num_scales', default=8, type=int)
parser.add_argument('--beta', default=4, type=float)

commandline_params = vars(parser.parse_args())
model_choice = commandline_params['vae_model']

trans_array = np.loadtxt("transformations.txt")
number_of_transformations = [commandline_params['num_rotations'], commandline_params['num_translation_x'], commandline_params['num_translation_y'], commandline_params['num_scales']]

params = {}
params['z_size'] = 20
params['beta'] = commandline_params['beta']
params['batch_size'] = np.sum(np.dot(trans_array, number_of_transformations), dtype=np.int)
if model_choice == 'mlp':
    params['X_size'] = 1600
    params['hidden_enc_1_size'] = 500
    params['hidden_enc_2_size'] = 200
    params['z_size'] = 20
    params['hidden_gen_1_size'] = 200
    params['hidden_gen_2_size'] = 500


thetas = np.linspace(-60, 60, commandline_params['num_rotations'])
trans_x = np.linspace(-6, 6, commandline_params['num_translation_x'])
trans_y = np.linspace(-6, 6, commandline_params['num_translation_y'])
scales = np.linspace(0.6, 1.5, commandline_params['num_scales'])


plt.style.use('ggplot')
if (trans_array[0]!=0):
    plt.figure()
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([-0.2,0.2])
        plt.plot(thetas, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('rotation angle in degrees')
    plt.savefig('rotation_mean_beta_{:}.png'.format(params['beta']))

if (trans_array[1]!=0):
    plt.figure()
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([-0.2,0.2])
        plt.plot(trans_x, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('translation_x in pixels')
    plt.savefig('translation_x_mean_beta_{:}.png'.format(params['beta']))

if (trans_array[2]!=0):
    plt.figure()
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([-0.2,0.2])
        plt.plot(trans_y, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('translation_y in pixels')
    plt.savefig('translation_y_mean_beta_{:}.png'.format(params['beta']))

if (trans_array[3]!=0):
    plt.figure()
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([-0.2,0.2])
        plt.plot(scales, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('scale')
    plt.savefig('scale_mean_beta_{:}.png'.format(params['beta']))

plt.figure(2)
plt.ylim([0,1.2])
plt.plot(mean_of_std_devs)
plt.savefig('beta_{:}.png'.format(params['beta']))
plt.show()

print('Program Finished')
