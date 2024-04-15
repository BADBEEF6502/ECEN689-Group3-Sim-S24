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
epoch = 1
batch_count = 100
batch_size = 500

# Equilibrium propagation parameters
alpha_1 = 0.01
alpha_2 = 0.005
beta = 1


# Free phase is calculated with beta set to zero, AKA ignore beta and mean square error
def free_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights):
    hidden_old = hidden_layer
    output_old = output_layer
    for _ in range(100):
        # Continually calculate dE/dW_ij and hidden layer until convergence (*.T means transpose)
        dEdW_ij = sigmoid_prime(hidden_old) * (np.dot(input_layer, in_hidden_weights) + np.dot(output_old, hidden_out_weights.T)) - hidden_old
        dEdW_ij2 = sigmoid_prime(output_old) * (np.dot(hidden_old, hidden_out_weights)) - output_old

        # Update the layers
        hidden_old = sigmoid(hidden_old + learning_rate*dEdW_ij)
        output_old = sigmoid(output_old + learning_rate*dEdW_ij2)
    
    # Return the calculated hidden and output layers
    return (hidden_old, output_old)




def clamped_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights, target):
    hidden_old = hidden_layer
    output_old = output_layer
    for _ in range(20):
        hidden_free = sigmoid_prime(hidden_old) * (np.dot(input_layer, in_hidden_weights) + np.dot(output_layer, hidden_out_weights.T)) - hidden_old
        output_free = sigmoid_prime(output_old) * (np.dot(hidden_layer, hidden_out_weights)) - output_old
        dCdW_ij = output_free + beta * (target - output_old)

        # Update layers
        hidden_old = hidden_free
        output_old = sigmoid(output_old + learning_rate*dCdW_ij)


    return (hidden_old, output_old)


# Save total correct guesses over time
correct_guess = 0
guess_mat = np.zeros((batch_count * batch_size * epoch))

# Start an epoch
for e in range(epoch):

    # Start a batch
    for b in range(batch_count):

        # Start an iteration
        for iteration in range(batch_size):        
            # Set the input layer
            input_layer = data[500*b + iteration]

            # Generate the target vector
            target = np.zeros(10)
            for i in range(output_size):
                if (i == label[500*b + iteration]):
                    target[i] = 1
                else:
                    target[i] = 0
            
            # Calculate free and weakly clamped phase
            h_free, out_free = free_phase(input_layer, hidden_layer, output_layer, in_hidden_weights, hidden_out_weights)
            h_clamped, out_clamped = clamped_phase(input_layer, h_free, out_free, in_hidden_weights, hidden_out_weights, target)

            # Determine if the guess is correct using the free phase
            if (np.argmax(target) == np.argmax(out_free)):
                correct_guess += 1
                guess_mat[e*50000 + 500*b + iteration] = correct_guess

            # Calculate change in theta, AKA actual change to weights
            dW = alpha_1 * (1 / beta) * (np.outer(input_layer, h_clamped) - np.outer(input_layer, h_free))
            dW2 = alpha_2 * (1 / beta) * (np.outer(hidden_layer, out_clamped) - np.outer(hidden_layer, out_free))

            # Calculate new weights and run again
            in_hidden_weights += dW
            hidden_out_weights += dW2

            # print out mse
            if (b*500 + iteration) % 100 == 0:
                print("iteration #" + str(b*500 + iteration), np.square(target - out_free).mean())

            # Diminish alpha over time
            if ((b*500 + iteration) % 2500 == 2499):
                alpha_1 /= 10
                alpha_2 /= 10


# Graph the correct guesses over time
plt.plot(range(batch_count*batch_size*epoch), guess_mat)
plt.savefig("plot.png")
# plt.show()
