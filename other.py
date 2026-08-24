import numpy as np
inputs = [[1, 2, 3 ,2.5], 
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-.8]]


weights = [[.2, .8, -.5, 1],
           [0.5,-0.91,.26,-.5],
           [-.26,-.27,.17,.87]]


weights2 = [[.1,-.14,.5],
           [-.5,.12,-.33],
           [-.44,.73,-.13]]

biases = [2, 3, .5]
biases2 = [-1, 2, -.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)