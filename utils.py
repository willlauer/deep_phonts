import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import imageio
import numpy as np

from hyper_params import params



def load_emnist(split, needs_download, batch_size_train=1000, batch_size_test=1000):


	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.EMNIST('./data/', split, train=True, download=needs_download,
									transform=torchvision.transforms.Compose([
									torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize(
										(0.1307,), (0.3081,))
									])),
		batch_size=batch_size_train, shuffle=True)

	val_loader = torch.utils.data.DataLoader(
		torchvision.datasets.EMNIST('./data/', split, train=False, download=needs_download,
									transform=torchvision.transforms.Compose([
									torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize(
										(0.1307,), (0.3081,))
									])),
		batch_size=batch_size_test, shuffle=True)


	return train_loader, val_loader


def visualize_samples(loader):

	examples = enumerate(loader)
	batch_idx, (example_data, example_targets) = next(examples)

	fig = plt.figure()
	for i in range(6):
		plt.subplot(2,3,i+1)
		plt.tight_layout()
		plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
		plt.title('Ground truth: {}'.format(example_targets[i]))
		#print(example_data[i][0])
		plt.xticks([])
		plt.yticks([])
	plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def visualize_filters(weight):
    '''
    Todo: clean up or remove
    '''
    grid = visualize_grid(weight.transpose(0, 2, 3, 1))
    print(grid.shape)
    plt.imshow(grid.astype('uint8').squeeze())
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()

def visualize_weights(model):

    # 150 and 180 can be changed for more or less noise
    img = np.uint8(np.random.uniform(150, 180, (28, 28, 3)))/255

    img_var = Variable(img[None], requires_grad=True)


	# Use random noise to compute the activation at different layers

	#noise = torch.randint(0, 256, (28,28))


def postprocess_image(img_path, threshold=115):
	'''
	given grayscale numpy representation of image, based on given threshold
	will max or min out grayscale pixel values to better visualize result
	'''
	# TODO possibly make this better with gradients for edges/curves
	img = imageio.imread(img_path)
	img = np.where(img < threshold, 0, 255)
	imageio.imsave(img_path[:-4] + '_post' + img_path[-4:], img)
	return img
