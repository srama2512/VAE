import argparse
import matplotlib.pyplot as plt
import numpy as np

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

z_mean_theta = np.load("z_mean_theta_beta_{:}.npy".format(commandline_params['beta']))
mean_of_std_devs = np.load("mean_of_std_devs_beta_{:}.npy".format(commandline_params['beta']))
z_var = np.load("z_var_beta_{:}.npy".format(commandline_params['beta']))


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
    plt.figure(figsize=(13.66,7.68))
    plt.clf()
    gmin, gmax = np.min(z_mean_theta), np.max(z_mean_theta)
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([gmin, gmax])
        plt.plot(thetas, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('rotation angle in degrees')
    plt.suptitle('Latent Variable means across rotations. Beta={:}'.format(commandline_params['beta']))
    plt.tight_layout()
    plt.savefig('rotation_mean_beta_{:}.png'.format(params['beta']), dpi=200)

if (trans_array[1]!=0):
    plt.figure(figsize=(13.66,7.68))
    plt.clf()
    gmin, gmax = np.min(z_mean_theta), np.max(z_mean_theta)
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([gmin, gmax])
        plt.plot(trans_x, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('translation_x in pixels')
    plt.suptitle('Latent Variable means across x translations. Beta={:}'.format(commandline_params['beta']))
    plt.tight_layout()
    plt.savefig('translation_x_mean_beta_{:}.png'.format(params['beta']), dpi=200)

if (trans_array[2]!=0):
    plt.figure(figsize=(13.66,7.68))
    plt.clf()
    gmin, gmax = np.min(z_mean_theta), np.max(z_mean_theta)
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([gmin, gmax])
        plt.plot(trans_y, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('translation_y in pixels')
    plt.subptitle('Latent Variable means across y translations. Beta={:}'.format(commandline_params['beta']))
    plt.tight_layout()
    plt.savefig('translation_y_mean_beta_{:}.png'.format(params['beta']), dpi=200)

if (trans_array[3]!=0):
    plt.figure(figsize=(13.66,7.68))
    plt.clf()
    gmin, gmax = np.min(z_mean_theta), np.max(z_mean_theta)
    for i in range(params['z_size']):
        z_across_transformation = z_mean_theta[:, i]
        plt.subplot(params['z_size']/4, 4, i+1)
        plt.ylim([gmin, gmax])
        plt.plot(scales, z_across_transformation)
        plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

    plt.xlabel('scale')
    plt.suptitle('Latent Variable means across scales. Beta={:}'.format(commandline_params['beta']))
    plt.tight_layout()
    plt.savefig('scale_mean_beta_{:}.png'.format(params['beta']), dpi=200)

plt.figure(figsize=(13.66,7.68))
plt.clf()
gmin, gmax = np.min(mean_of_std_devs), np.max(mean_of_std_devs)
plt.ylim([gmin, gmax])
plt.title('Std Dev averaged over data. Beta={:}'.format(commandline_params['beta']))
plt.ylabel('Avg Std Dev')
plt.xlabel('Latent Variable number')
plt.plot(range(params['z_size']), mean_of_std_devs)
plt.tight_layout()
plt.savefig('avg_stddev_beta_{:}.png'.format(params['beta']))
# plt.show()

print('Program Finished')
