import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))

def sigmoid_prime(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))


# Free phase is calculated with beta set to zero, AKA ignore beta and mean square error
def free_phase(X, batch_size, hidden_weights, hidden_size, out_weights, output_size, learning_rate):
    hidden_old = np.random.randn(batch_size, hidden_size)
    output_old = np.random.randn(batch_size, output_size)
    for i in range(100):
        # Continually calculate dE/dW_ij and hidden layer until convergence (*.T means transpose)
        dEdW_ij = sigmoid_prime(hidden_old) * ((X@hidden_weights) + (output_old@out_weights.T)) - hidden_old
        dEdW_ij2 = sigmoid_prime(output_old) * (hidden_old@out_weights) - output_old
        
        # Update the layers
        hidden_old = sigmoid(hidden_old + learning_rate*dEdW_ij)
        output_old = sigmoid(output_old + learning_rate*dEdW_ij2)
    
    # Return the calculated hidden and output layers
    return (hidden_old, output_old)




def clamped_phase(X, batch_size, hidden_weights, hidden_size, out_weights, output_size, learning_rate, y):
    hidden_old = np.random.randn(batch_size, hidden_size)
    output_old = np.random.randn(batch_size, output_size)
    for i in range(20):
        hidden_free = sigmoid_prime(hidden_old) * ( X@hidden_weights + output_layer@out_weights.T) - hidden_old
        output_free = sigmoid_prime(output_old) * (hidden_layer@out_weights) - output_old
        dCdW_ij = output_free + beta * (y - output_old)

        # Update layers
        hidden_old = hidden_free
        output_old = sigmoid(output_old + learning_rate*dCdW_ij)


    return (hidden_old, output_old)

def train_net(epocs, batch_size, data, label, hidden_weights,hidden_size, out_weights, output_size, alpha_1, alpha_2, beta, learning_rate):
    #break if batch size isnt  even (simplifys code)
    if(np.shape(data)[0]%batch_size!=0):
        raise Exception("data must be divisible by batch size!")
    
    #starts epoc
    for epoc in range(epocs):
        #starts batch
        for batch in range(np.shape(data)[0]//batch_size):
            #grabs X and Y data
            X = data[batch*batch_size:(batch+1)*batch_size]
            y0 = label[batch*batch_size:(batch+1)*batch_size]
            y = np.zeros((y0.size, 10))
            y[np.arange(y0.size), y0] = 1
            #print(y)

            
            #gets U of free and clamped phase
            h_free, out_free = free_phase(X, batch_size, hidden_weights,hidden_size, out_weights,output_size, learning_rate)
            h_clamped, out_clamped = clamped_phase(X, batch_size, hidden_weights,hidden_size, out_weights,output_size, learning_rate, y)
            
            #calcualtes dW
            dW = alpha_1 * (1 / beta)* (1/1) *(1/batch_size)* ((X).T@(h_clamped) - (X).T@(h_free))
            dW2 = alpha_2 * (1 / beta) * (1/1) *(1/batch_size)* ((h_clamped).T@(out_clamped) - (h_clamped).T@(out_free))
            
            #updates W
            hidden_weights += dW
            out_weights += dW2
            
            # Determine if the guess is correct using the free phase
            # USE FORWARD PROP
            if(batch%50==0):
                h = sigmoid(X @ hidden_weights)
                out = sigmoid(h @ out_weights)
                print(np.argmax(out,1)-np.argmax(y,1))
                mse = np.sum((np.argmax(out,1)==np.argmax(y,1)))
                print('batch num:', batch ,' accuracy:', mse/batch_size)
            
            

            
    print('done')

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
hidden_weights = np.random.randn(input_size, hidden_size)
out_weights = np.random.randn(hidden_size, output_size)

# Load in the input data
mnist_data = np.load('sim/MNIST_8x8.npz')
label = mnist_data['x']
data = mnist_data['y']


# Initialize variables related to model generation
epoch = 10
batch_count = 100
batch_size = 100

# Equilibrium propagation parameters
alpha_1 = 0.01
alpha_2 = 0.005
beta = 1
data, x_test, label, y_test = train_test_split(data, label, test_size=0.1)

train_net(epoch, batch_size, data, label, hidden_weights, hidden_size, out_weights, output_size, alpha_1, alpha_2, beta, learning_rate )
