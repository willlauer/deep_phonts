import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

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


def visualize_weights(path):

	model = torch.load(path)

	# Use random noise to compute the activation at different layers
	
	noise = torch.randint(0, 256, (28,28))


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




def run_postprocess():
	"""
	In the event that we forget or decide to not use the postprocessing as part of the main script, we can
	also run this method from the command line
	:return: None
	"""
	for image in os.listdir('./transfer_checkpoint_images'):
		img_path = './transfer_checkpoint_images/' + image
		_ = postprocess_image(img_path)