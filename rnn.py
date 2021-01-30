import numpy as np
from rnn_utils import *


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Waa -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax =  parameters["Wax"]
    Waa =  parameters["Waa"]
    Wya =  parameters["Wya"]
    ba =  parameters["ba"]
    by =  parameters["by"]
    
    
    # Compute the next activation
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    
    
    # Compute prediction at time step "t"
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    
    
    # tuple of values needed for the backward pass
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    # Retrive the dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    
    # Initialize a which include the hidden states for every time-step
    a = np.zeros((n_a, m , T_x))
    
    # Initialize y_pred which includes predictions for every time-step
    y_pred = np.zeros((n_y, m , T_x))
    
    # Initialize caches which includes the list of caches 
    caches = []
    
    
    # Loop over all time steps
    for t in range(T_x):
        
        # Retrive xt (input for the time step of t)
        xt = x[:,:,t]
        
        #Update next hidden state, compute the prediction, get the cache
        a[:,:,t], y_pred[:,:,t], cache = rnn_cell_forward(xt, a0, parameters)
        
        # Update a0
        a0 = a[:,:,t]
        
        # Append "cache" to "caches"
        caches.append(cache)
        
    # store values needed for backward propagation in cache
    caches = (caches, x) 
    
    return a, y_pred, caches

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt = your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev = Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev = Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters = python dictionary containing:
                        Wf = Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf = Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi = Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi = Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc = Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc =  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo = Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo =  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy = Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by = Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next = next hidden state, of shape (n_a, m)
    c_next = next memory state, of shape (n_a, m)
    yt_pred = prediction at timestep "t", numpy array of shape (n_y, m)
    cache = tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """
    
    # Concate xt and a_prev
    concat = np.concatenate((a_prev, xt), axis = 0)
    
    # Retrive the parameters
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']
    
    # Calculate candidate value (c tilde)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    
    # Forget gate
    ft = sigmoid(np.dot(Wf, concat) + bf)
    
    # Update gate
    it = sigmoid(np.dot(Wi, concat) + bi)
    
    # Output gate
    ot = sigmoid(np.dot(Wo, concat) + bo)
    
    # Compute next memory state
    c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
    
    # Compute next hidden state
    a_next = np.multiply(ot, np.tanh(c_next))
    
    # Compute yt_pred
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    
    
    # values needed for the backward pass 
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    
    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    # Initialize chaches
    caches = []
    
    # Retrive the dimmensions
    (n_x, m, T_x) = x.shape
    Wy = parameters['Wy']
    (n_y, n_a) = Wy.shape
    
    
    # Initialize a
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    
    # Initialize y
    y = np.zeros((n_y, m, T_x))
    
    # Initialize c0
    
    c0 = np.zeros((n_a, m))
    
    # Loop over different time step
    for t in range(T_x):
        
        # Forward single pass
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t], a0, c0, parameters)
        
        # Store a_next
        a[:,:,t] = a_next
        
        # Store c_next
        c[:,:,t] = c_next
        
        # Store yt_pred
        y[:,:,t] = yt_pred
        
        
        # Append cache
        caches.append(cache)
        
        
        # Update a0 and x_prev
        a0 = a_next
        c0 = c_next
        
    caches = (caches, x)
    
    
    return a, y, c,  caches


def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    # Retrive the information
    a_next, a_prev, xt, parameters = cache
    
    # Retrieve parameters from "parameters"
    Wax =  parameters["Wax"]
    Waa =  parameters["Waa"]
    Waa =  parameters["Waa"]
    ba =  parameters["ba"]
    by =  parameters["by"]
    
    dtanh = np.multiply(1 - np.square(np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)), da_next)
    
    dxt = np.dot(Wax.T, dtanh)
    
    
    da_prev = np.dot(Waa.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis = 1).reshape(-1, 1)
    
    
    gradients = {'dxt':dxt, 'da_prev':da_prev, 'dWax':dWax, 'dWaa':dWaa, 'dba':dba}
    
    
    return gradients

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    ### START CODE HERE ###
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    
    return a_next, p_t

# def rnn_forward(X, Y, a_prev, parameters, vocab_size = 27):
#     """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
#     It returns the loss' value as well as a "cache" storing values to be used in the backpropagation.
#     """
#     # Initialize x, a, and y_hat as empty dictionaries
#     x, a, y_hat = {}, {}, {}
    
#     a[-1] = np.copy(a_prev)
    
#     # Initialize the loss to 0
#     loss = 0 
    
#     for t in range(len(X)):
        
#         # Set x[t] to be the one-hot vector representation of the t'th character in X.
#         x[t] = np.zeros((vocab_size,1)) 
#         if (X[t] != None):
#             x[t][X[t]] = 1
            
#         # Run one step forward of the RNN
#         a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
#         # Update the loss by substracting the cross-entropy term of this time-step from it.
#         loss -= np.log(y_hat[t][Y[t], 0])
        
#     cache = (y_hat, a, x)
        
#     return loss, cache


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    ### END CODE HERE ###
    
    return gradients, a

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- 

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation).
    x = np.zeros((vocab_size, 1))
    
    #Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))
    
    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
    indices = []
    
    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1 
    
    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
    # trained model), which helps debugging and prevents entering an infinite loop. 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        # Step 2: Forward propagate x
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        np.random.seed(counter + seed) 
        
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        
        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        seed += 1
        counter += 1
        
    if (counter == 50):
        indices.append(char_to_ix['\n'])
          
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    ### START CODE HERE ###
    
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    ### END CODE HERE ###
    
    return loss, gradients, a[len(X)-1]