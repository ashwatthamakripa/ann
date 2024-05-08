import numpy as np


v1 = np.array([1, -1, 1, -1])
v2 = np.array([-1, 1, -1, 1])
v3 = np.array([1, 1, -1, -1])
v4 = np.array([-1, -1, 1, 1])


w = np.outer(v1, v1) + np.outer(v2, v2) + np.outer(v3, v3) + np.outer(v4, v4) 


np.fill_diagonal(w, 0)


def activation(x):
    return np.where(x >= 0, 1, -1)


def hopfield_network(input_vector, weight_matrix, activation_function, iterations):
    
    state = input_vector
    
    for i in range(iterations):
        
        net_input = np.dot(weight_matrix, state)
        
        new_state = activation_function(net_input)
        
        if np.array_equal(new_state, state):
            
            return new_state
        
        state = new_state
    
    return state


input_vector = np.array([-1, -1, 1, -1])
output_vector = hopfield_network(input_vector, w, activation, 10)
print("Input vector:", input_vector)
print("Output vector:", output_vector)
w