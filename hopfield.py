import numpy as np



def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))

def sigmoid_prime(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))



# Initialize network size
input_size = 8 * 8
hidden_size = 32
output_size = 10

# Initialize input and output layers
input_layer = np.zeros((input_size, 1))
hidden_layer = np.zeros((hidden_size, 1))
output_layer = np.zeros((output_size, 1))

# Also initialize weights matrices, but with random numbers
in_hidden_weights = np.random.randn(input_size, hidden_size)
hidden_out_weights = np.random.randn(hidden_size, output_size)

# Load in the input data
mnist_data = np.load('sim/MNIST_8x8.npz')
label = mnist_data['x']
data = mnist_data['y']

# Initialize variables related to model generation
epoch = 10
batch_count = 100
batch_size = 500



# Start an epoch
for e in range(epoch):

    # Start a batch
    for b in range(batch_count):

        # Start an iteration
        for iteration in range(batch_size):         

            # Set input to be next image
            input_layer = data[iteration*b]
        
            # Take in the input and multiply by a weights matrix
            hidden_layer = input_layer @ in_hidden_weights
        
            # Run through sigmoid function
            hidden_layer = sigmoid(hidden_layer)
        
            # Run new matrix through second weights set into output
            output_layer = hidden_layer @ hidden_out_weights

            # Run through sigmoid function
            output_layer = sigmoid(output_layer)
        
            # Setup equilibrium propagation to update weights
            



