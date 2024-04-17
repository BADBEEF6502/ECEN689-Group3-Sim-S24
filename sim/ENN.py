import numpy as np

# Load in the MNIST data.
mnist_data = np.load('MNIST_8x8.npz')
labels = mnist_data['x']
targets = mnist_data['y']

# Network definition.
INPUT_SIZE  = 64
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10

# Network Hyperparameters.
MAX_EPOCH                = 50000    # Maximum training epoch.
MAX_FREE_PHASE           = 100      # Must be experimentaly determined.
MAX_WEAKLY_CLAMPED_PHASE = 20       # Must be experimentaly determined.
EVAL                     = 100      # Every 100 steps perform an evaluation.
BETA                     = 1.0      # Default.
EPSILON                  = 0.05     # Exploration hyperparameter.
alpha1                   = 0.1      # Alpha1 is a learning rate hyperparameter, adjustable in run.
alpha2                   = 0.05     # Alpha2 is a learning rate hyperparameter, adjustable in run.

# One hot encoding for each target value, accessed oneHotEncode[target].
oneHotEncode = np.eye(OUTPUT_SIZE)

# Weight init
wxh = np.random.rand(INPUT_SIZE, HIDDEN_SIZE)   # Weights Input (x) and Hidden.
why = np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE)  # Weights Hidden (h) and Output (y).
bh  = np.random.rand(HIDDEN_SIZE)               # Bias for Hidden (h).
by  = np.random.rand(OUTPUT_SIZE)               # Bias for Output (y).

# The activation function must be differentiable, sigmoid (just as in the paper) is used here.
# Input: u which is the value of the neuron before activation function applied.
def sig(u):
    x = np.clip(u, -1.0, 1.0)                   # Clip is applied here for numerical stability.
    return np.copy(1.0 / (1.0 + np.exp(-x)))    # np.copy() is used to ensure deep copy.

# Derivative of sigmoid function.
def sigPrime(u):
    return np.copy(sig(u) * (1.0 - sig(u)))

# Free Phase. 
def freePhase(h, x, y, wxh, why, by, bh):
    gh = sigPrime(h) * (np.dot(x, wxh) + np.dot(y, why.T) + bh) - h     # Gradient for hidden layer.
    gy = sigPrime(y) * (np.dot(h, why) + by) - y                        # Gradient for output layer.
    return np.copy(gh), np.copy(gy)

# Weakly Clamped Phase.
def weaklyClampedPhase(h, x, y, wxh, why, by, bh, BETA, t):
    gh = sigPrime(h) * (np.dot(x, wxh) + np.dot(y, why.T) + bh) - h     # Gradient for hidden layer.
    gy = sigPrime(y) * (np.dot(h, why) + by) - y + BETA * (t - y)       # Gradient for output layer.
    return np.copy(gh), np.copy(gy)

# Used for debugging and data collection.
correct = 0                     # Cumulative result of correct guesses per epoch.
guesses           = np.empty(MAX_EPOCH) # Store all of the guesses.
actual            = np.empty(MAX_EPOCH) # Store all of the target results.
accuracy_overtime = []                  # Store all of the guesses.
mse_overtime      = []                  # Store all of the target results.

assert MAX_EPOCH % EVAL == 0, 'ERROR: MAX_EPOCH MUST BE DIVISABLE BY EVAL!'

# --- MAIN RUN ---
for epoch in range(MAX_EPOCH):
    selected = np.random.randint(0, len(targets))   # Select a single target at a time, randomly select from entire MNIST population.
    x = targets[selected]                           # Raw input as a vector for each number.
    t = oneHotEncode[labels[selected]]              # Target is oneHotEncoded for correct label.

    # Layer init
    h = np.random.rand(HIDDEN_SIZE) # Hidden layer activations.
    y = np.random.rand(OUTPUT_SIZE) # Output layer activations.

    # Free phase computation.
    for _ in range(MAX_FREE_PHASE):
        gh, gy = freePhase(h, x, y, wxh, why, by, bh)
        h = sig(h + EPSILON * gh)
        y = sig(y + EPSILON * gy)

    hf = np.copy(h)  # Hidden free equilibrium state.
    yf = np.copy(y)  # Output free equilibrium state.

    # Weakly clamped phase computation.
    for _ in range(MAX_WEAKLY_CLAMPED_PHASE):
        gh, gy = weaklyClampedPhase(h, x, y, wxh, why, by, bh, BETA, t)
        h = sig(h + EPSILON * gh)
        y = sig(y + EPSILON * gy)

    hc = np.copy(h) # Hidden Weakly clamped phase equilibrium state.
    yc = np.copy(y) # Output Weakly clamped phase equilibrium state.

    # Update weight matricies.
    wxh += alpha1 * (1.0 / BETA) * (np.outer(x, hc) - np.outer(x, hf))
    why += alpha2 * (1.0 / BETA) * (np.outer(hc, yc) - np.outer(hf, yf))

    # Determine if guess was correct.
    guess = np.argmax(yf)                               # Get the networks guess for what the number is.
    correct += np.array_equal(t, oneHotEncode[guess])   # Returns 'True' 1 if correct or 'False' 0 if incorrect.
    guesses[epoch] = guess
    actual[epoch] = labels[selected]
    
    # Display epoch and cumulative count correct.
    if epoch % EVAL == 0 and epoch != 0:
        mse = (1.0 / OUTPUT_SIZE) * np.sum((yf - t) ** 2)   # Using formula: 1/N * sum((model - target) ** 2)
        accuracy_per_epoch = correct / EVAL
        accuracy_overtime.append(accuracy_per_epoch)
        mse_overtime.append(mse)
        correct = 0                                         # Reset correct per epoch to 0.
        print(f'{epoch}\t{round(accuracy_per_epoch, 3)}\t{round(mse, 3)}')

# Record the performance in a tsv.
with open('ENN_perf.tsv', 'w') as f:
    print('Epoch\tAccuracy\tMSE', file=f)
    for i in range(MAX_EPOCH // EVAL):
        print(f'{i * EVAL}\t{accuracy_overtime[i]}\t{mse_overtime[i]}', file=f)

# Record the results in a tsv.
with open('ENN_results.tsv', 'w') as f:
    print('Iteration\tGuess\tActual\tCorrect?', file=f)
    for i in range(MAX_EPOCH):
        print(f'{i}\t{guesses[i]}\t{actual[i]}\t{guesses[i] == actual[i]}', file=f)

exit()