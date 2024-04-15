import numpy as np
import matplotlib.pyplot as plt


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
input_layer = np.zeros((1, input_size))
hidden_layer = np.zeros((1, hidden_size))
output_layer = np.zeros((1, output_size))

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
alpha_1 = 0.01
alpha_2 = 0.005
beta = 1


# Free phase is calculated with beta set to zero, AKA ignore beta and mean square error
def free_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights):
    for _ in range(100):
        # Continually calculate dE/dW_ij and hidden layer until convergence (*.T means transpose)
        dEdW_ij = sigmoid_prime(hidden_layer) * (np.dot(input_layer, in_hidden_weights) + np.dot(output_layer, hidden_out_weights.T)) - hidden_layer
        dEdW_ij2 = sigmoid_prime(output_layer) * (np.dot(hidden_layer, hidden_out_weights)) - output_layer

        # Update the layers
        hidden_layer = sigmoid(hidden_layer + learning_rate*dEdW_ij)
        output_layer = sigmoid(output_layer + learning_rate*dEdW_ij2)
    
    # Return the calculated hidden and output layers
    return (hidden_layer, output_layer)




def clamped_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights, target):
    for i in range(20):
        hidden_free, output_free = free_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights)
        dCdW_ij = output_free + beta * (target - output_layer)

        # Update layers
        hidden_layer = hidden_free
        output_layer = sigmoid(output_layer + learning_rate*dCdW_ij)


    return (hidden_layer, output_layer)


# Save total correct guesses over time
correct_guess = 0
guess_mat = np.zeros((batch_count * batch_size))

# Start an epoch
for e in range(epoch):

    # Start a batch
    for b in range(batch_count):

        # Start an iteration
        for iteration in range(batch_size):         
            # Generate the target vector
            target = np.zeros(10)
            for i in range(10):
                if (i == label[iteration*b]):
                    target[i] = 1
                else:
                    target[i] = 0
            
            # Calculate free and weakly clamped phase
            h_free, out_free = free_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights)
            h_clamped, out_clamped = clamped_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights, target)

            # Determine if the guess is correct using the free phase
            if (np.argmax(target) == np.argmax(out_free)):
                correct_guess += 1
                guess_mat[500*b + iteration] = correct_guess

            # Calculate change in theta, AKA actual change to weights
            dW = alpha_1 * (1 / beta) * (np.outer(input_layer, h_clamped) - np.outer(input_layer, h_free))
            dW2 = alpha_2 * (1 / beta) * (np.outer(hidden_layer, out_clamped) - np.outer(hidden_layer, out_free))

            # Calculate new weights and run again
            in_hidden_weights += dW
            hidden_out_weights += dW2

            # Diminish alpha over time
            if ((b*500 + iteration) % 2500 == 2499):
                alpha_1 /= 10
                alpha_2 /= 10


# Graph the correct guesses over time
plt.plot(range(batch_count*batch_size), guess_mat)
plt.savefig("plot.png")
# plt.show()
