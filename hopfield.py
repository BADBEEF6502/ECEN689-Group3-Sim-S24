import numpy as np



def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))

def sigmoid_prime(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))



# Initialize network size
input_size = 8 * 8
hidden_size = 32
output_size = 10

# Add learning rate
learning_rate = 0.1

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

# Equilibrium propagation parameters
beta = 1



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
            hidden_layer_sig = sigmoid(hidden_layer)
        
            # Run new matrix through second weights set into output
            output_layer = hidden_layer_sig @ hidden_out_weights

            # Run through sigmoid function
            output_layer_sig = sigmoid(output_layer)

            # Generate the target vector
            target = np.zeros(10)
            for i in range(10):
                if (i == label[iteration*b]):
                    target[i] = 1
                else:
                    target[i] = 0
        
            # Setup equilibrium propagation to update weights
                    
            # First, calculate the output derivative of Hebbian energy matrix
            dEdW_ij = np.zeros(output_size, 10)
            for i in range(output_size):
                for j in range(output_size):
                    if (i != j):
                        dEdW_ij[i] += -0.5 * (output_layer_sig[i] * output_layer_sig[j])

            # Next, calculate the mean square error derivative matrix
            # Chain rule, pain rule
            dCdo = - (target - output_layer_sig)
            dody = sigmoid_prime(output_layer)

            # Fill based on inputs from hidden layer, (hidden_layer_sig)
            dydW_ij = np.zeros((output_size, hidden_size))
            for i in range(output_size):
                for j in range(hidden_size):
                    pass
                    # Sum up the inputs that are multiplied into the weights matrix indicated by i and j
                    dydW_ij[i][j] += hidden_layer_sig[j]

            # Set G2 gradient matrix to be element-wise multiplication of dCdo * dody and matrix multiply dydW_ij
            G2 = np.multiply(dCdo, dody) @ dydW_ij

            # Now set up gradient matrix G1 for use in recalculating the gradient
            dCdo2 = np.multiply(hidden_layer_sig, dEdW_ij)
            dody2 = sigmoid_prime(hidden_layer)
            dydW_ij2 = np.zeros((hidden_size, input_size))
            for i in range(hidden_size):
                for j in range(input_size):
                    dydW_ij2[i][j] += input_layer[j]
            
            G1 = np.multiply(dCdo2, dody2) @ dydW_ij2

            # Apply the gradient and learning rate to the weights and try again
            in_hidden_weights -= np.multiply(learning_rate, G1)
            hidden_out_weights -= np.multiply(learning_rate, G2)


