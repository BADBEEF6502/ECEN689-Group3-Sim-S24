import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


# Setup an array to store free phase and total energy values
total_energy = []
free_energy = []
clamped_energy = []

#sigmod
def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))

def sigmoid_prime(matrix):
    return sigmoid(matrix) * (1 - sigmoid(matrix))


# Free phase is calculated with beta set to zero, AKA ignore beta and mean square error
def free_phase(X, batch_size, hidden_weights, out_weights, learning_rate):
    hidden_old = np.random.randn(batch_size, hidden_weights.shape[1])
    output_old = np.random.randn(batch_size, out_weights.shape[1])
    for i in range(40):
        # Continually calculate dE/dW_ij and hidden layer until convergence (*.T means transpose)
        dHidden = sigmoid_prime(hidden_old) * ((X@hidden_weights) + (output_old@out_weights.T)) - hidden_old
        dOut = sigmoid_prime(output_old) * (hidden_old@out_weights ) - output_old
        
        #print(np.sum(dHidden))
        # Update the layers
        hidden_old = sigmoid(hidden_old + learning_rate*dHidden)
        output_old = sigmoid(output_old + learning_rate*dOut)
    
    # Calculate new values at hidden layer and output layer
    hidden_layer_values = sigmoid(X @ hidden_weights)
    output_layer_values = sigmoid(hidden_layer_values @ out_weights)
    # Load in the latest free phase energy value
    energy_val = 0.0
    for i in range(hidden_size):
        energy_val += (hidden_layer_values[0][i] * hidden_layer_values[0][i])
        for j in range(output_size):  
            if i != j:     
                energy_val -= out_weights[i][j] * hidden_layer_values[0][i] * output_layer_values[0][j]
    energy_val /= 2
    free_energy.append(energy_val)

    # Return the calculated hidden and output layers
    return (hidden_old, output_old)




def clamped_phase(X, hidden_old, output_old, hidden_weights, out_weights, learning_rate, y):

    for i in range(20):
        dHidden = sigmoid_prime(hidden_old) * ( X@hidden_weights + output_old@out_weights.T) - hidden_old
        dOut = sigmoid_prime(output_old) * (hidden_old@out_weights) - output_old
        dOut = dOut + beta * (y - output_old)

        # Update layers
        hidden_old = sigmoid(hidden_old + learning_rate*dHidden)
        output_old = sigmoid(output_old + learning_rate*dOut)
    
    # Create mean squared error here
    # Calculate new values at hidden layer and output layer
    hidden_layer_values = sigmoid(X @ hidden_weights)
    output_layer_values = sigmoid(hidden_layer_values @ out_weights)
    clamped_energy.append(0.5 * np.abs(y - output_layer_values).sum() * np.abs(y - output_layer_values).sum())

    return (hidden_old, output_old)

def train_net(epocs, batch_size, data, label, hidden_weights, out_weights, alpha_1, alpha_2, beta, learning_rate, X_test, y_test):
    #break if batch size isnt  even (simplifys code)
    if(np.shape(data)[0]%batch_size!=0):
        raise Exception("data must be divisible by batch size!")
    #finds number of iterations per batch
    iterations = np.shape(data)[0]//batch_size
    
    mse_list=np.zeros((epocs*iterations))
    accuracy_list=np.zeros((epocs*iterations))
    test_acc= np.zeros((epocs))
    test_acc_x = (np.arange(epocs)+1)*iterations
    mse_index=0
    
    #starts epoc
    for epoc in range(epocs):
        for batch in range(iterations):
            #grabs X and Y data
            X = data[batch*batch_size:(batch+1)*batch_size]
            y0 = label[batch*batch_size:(batch+1)*batch_size]
            y = np.zeros((y0.size, 10))
            y[np.arange(y0.size), y0] = 1
            #print(y)

            
            #gets U of free and clamped phase
            h_free, out_free = free_phase(X, batch_size, hidden_weights, out_weights, learning_rate)
            h_clamped, out_clamped = clamped_phase(X, h_free,out_free, hidden_weights, out_weights, learning_rate, y)
            
            # Determine if the guess is correct using the free phase
            # USE FORWARD PROP
            mse = (1.0 / batch_size) * np.sum((y - out_free) ** 2) 
            accuracy = np.sum((np.argmax(out_free,1)==np.argmax(y,1)))/batch_size
            mse_list[mse_index]=mse
            accuracy_list[mse_index]=accuracy
            mse_index+=1
            if(batch%50==0):
                #print(np.argmax(out_free,1),np.argmax(y,1))
                ##print(out)
                print('epoc', epoc,'batch num:', batch ,' accuracy:', accuracy, 'MSE',mse)
                
            
            dW4 = np.zeros((64,hidden_weights.shape[1]))
            dW5 = np.zeros((hidden_weights.shape[1],10))
            
            for i in range(batch_size):
                dW4+=(np.outer(X[i],h_clamped[i])-np.outer(X[i],h_free[i]))
                #dW6+=np.outer(X[i],h_clamped[i])-np.outer(X[i],h_free[i])
                dW5+=(np.outer(h_clamped[i],out_clamped[i])-np.outer(h_free[i],out_free[i]))
                
            #calcualtes dW
            dW4 = alpha_1 * (1 / beta) *(dW4 )
            dW5 = alpha_2 * (1 / beta) *(dW5)
            
            #updates W
            hidden_weights += dW4
            out_weights += dW5
            
        if(epoc==1 or epoc==3):
            alpha_1/=10
            alpha_2/=10

        test_acc[epoc]= test_net(X_test, y_test,hidden_weights ,out_weights)
            
            
            
            

    return (mse_list, accuracy_list, test_acc, test_acc_x) 
    print('done')
    

def test_net(X_test, y_test, W1, W2):
    y = np.zeros((y_test.size, 10))
    y[np.arange(y_test.size), y_test] = 1
    batch_size = X_test.shape[0]
    h_free, out_free = free_phase(X_test, batch_size, W1, W2, 1)
    
    accuracy = np.sum((np.argmax(out_free,1)==np.argmax(y,1)))/batch_size
    return accuracy

# Initialize network size
input_size = 8 * 8
hidden_size = 64
output_size = 10

# Add learning rate
learning_rate = 1

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
epoch = 6
batch_count = 100
batch_size = 100

# Equilibrium propagation parameters
alpha_1 = 0.1
alpha_2 = 0.05
beta = 1
data, x_test, label, y_test = train_test_split(data, label, test_size=0.1)

start = time.time()
mse_list,accuracy_list, test_acc, test_acc_x = train_net(epoch, batch_size, data, label, hidden_weights, out_weights, alpha_1, alpha_2, beta, learning_rate, x_test, y_test )
end = time.time()
print(end-start, "second runtime")

# Generate the free energy plot
plt.plot(range(len(free_energy)), free_energy)
plt.title("Free Energy over time")
plt.xlabel("Episode #")
plt.ylabel("Free Energy")
plt.savefig("free_energy.png")
plt.close()

# Generate the free energy plot with averaging filter applied
avg_filter = np.ones(25) / 25
plt.plot(range(len(free_energy)), np.convolve(free_energy, avg_filter, mode='same'))
plt.title("Free Energy over time")
plt.xlabel("Episode #")
plt.ylabel("Free Energy")
plt.savefig("free_energy_smoothed.png")
plt.close()

# Generate the clamped energy plot
plt.plot(range(len(clamped_energy)), clamped_energy)
plt.title("Clamped Energy over time")
plt.xlabel("Episode #")
plt.ylabel("Clamped Energy")
plt.savefig("clamped_energy.png")
plt.close()

# Generate the hopfield energy plot
# First, generate the hopfield energy list
for ei in range(len(clamped_energy)):
    total_energy.append(free_energy[ei] + beta * clamped_energy[ei])
plt.plot(range(len(total_energy)), total_energy)
plt.title("Hopfield Energy over time")
plt.xlabel("Episode #")
plt.ylabel("Hopfield Energy")
plt.savefig("hopfield_energy.png")
plt.close()

# Generate the hopfield energy plot for episode 500 and onward
plt.plot(range(len(total_energy)-500), total_energy[500:])
plt.title("Hopfield Energy over time (skipping to episode 500)")
plt.xlabel("Episode # after 500")
plt.ylabel("Hopfield Energy")
plt.savefig("hopfield_energy_skipping.png")
plt.close()

# Generate the hopfield energy plot for episode 500 and onward, with averaging filter as well
plt.plot(range(len(total_energy)-500), np.convolve(total_energy[500:], avg_filter, mode='same'))
plt.title("Hopfield Energy over time (skipping to episode 500)")
plt.xlabel("Episode # after 500")
plt.ylabel("Hopfield Energy")
plt.savefig("hopfield_energy_skipping_smoothed.png")
plt.close()

# fig, ax = plt.subplots()
# plt.title("MSE and accuracy of Equalibrium Prop")
# mse_x=np.arange(len(mse_list))

# MSE = ax.plot(mse_list, color='tab:blue', label='MSE')

# ax.tick_params(axis='y', labelcolor='tab:blue' )
# #ax.legend()
# ax2 = ax.twinx()
# acc=ax2.scatter(test_acc_x, test_acc, color='tab:green', label = 'Test Accuracy')
# ax2.plot(accuracy_list, color='tab:orange', label = 'Batch Accuracy')

# ax2.set_ylim(0,1)
# acc.set_zorder(20)
# ax2.tick_params(axis='y', labelcolor='tab:orange')
# #fig.legend(loc='upper right')
# plt.grid()
# plt.show()
