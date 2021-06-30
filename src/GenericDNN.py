import torch

class GenericDNN(torch.nn.Module):
	"""
	Simple, generic Deep Neural Network (DNN) with one input layer, one 
	hidden layer and an output layer, i.e. satisfies the criteria of being 
	a 'deep' network in the universal approximation theorem (UAT) sense.
	"""
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Args:
			input_size: shape of input - excluding batch_size
			hidden_size: shape of hidden - excluding batch_size
			output_size: shape of output - excluding batch_size
		"""
		super().__init__()
		self.fc1 = torch.nn.Linear(input_size, hidden_size)
		self.fc2 = torch.nn.Linear(hidden_size, output_size)
		self.softplus = torch.nn.Softplus()

	def forward(self, x):
		z = self.softplus(self.fc1(x)) + x
		return self.fc2(z)







