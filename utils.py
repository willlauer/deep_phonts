import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from hyper_params import params

class ContentLoss(nn.Module):
	"""
	From pytorch tutorial
	"""
	def __init__(self, target,):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input


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