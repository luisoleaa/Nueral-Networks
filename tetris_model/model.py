# Need to look into docs to see what actions give rewards and make a Neural Network that
# Learns to use these specific action to try to get the highest reward, then train it
# Also learn how to have multiple emulations running at once and training/learning at the same time.

import numpy as np

# Create a element class that initializes with a set of inputs and has a number of neurons set,
# This element is then fed forward, in this forward method we get the weights * a set of inputs + bias at that index of weights/biases
# after each calculation, store it in self.output and then use an activate function on self.output.


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        # self.game = gym_tetris.make('TetrisA-v3')

        # intialize weights and biases
        self.weights = (0.1 * np.random.randn(number_of_inputs, number_of_neurons))
        self.biases = np.zeros((1,number_of_neurons))

    def forward(self, inputs):
       self.inputs = inputs
       self.output = np.dot(inputs, self.weights) + self.biases # input weight is (64,240,256,3) weight shape (238,128)
       return self.output
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims= True )
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
class Relu:
    def forward(self, z):
        self.inputs = z
        self.output = np.maximum(0,z) # gets the inputs greater than 0
        return self.output
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0 # makes the values that are less than 0, equal to 0  
        return self.dvalues

class Adam:
    def __init__(self, params, lr = 2.5e-4, betas=(0.9, 0.999), eps = 1e-8):
        self.params = params # list of ndarrays, updated in place

        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in params] # initializes momentum to 0
        self.v = [np.zeros_like(p) for p in params] # initlaizes adaptive lr to 0
        self.t = 0 
    def step(self, grads):
        # Adam step formulas
        self.t +=1
        for i, (p,g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1- self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * (g*g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr *  mhat / (np.sqrt(vhat) + self.eps)


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis = 1, keepdims= True))
    return exp_logits / np.sum(exp_logits, axis = 1, keepdims= True)


# checking the relative error of feeding forward and backward between layers
def gradient_check(param, dparam, forward_fn, epsilon = 1e-5):
    numerical_grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + epsilon
        loss_plus = forward_fn()

        param[idx] = original_value - epsilon
        loss_minus = forward_fn()

        numerical_grad[idx] = (loss_plus - loss_minus) / (2*epsilon)
        it.iternext()

    numerator = np.abs(numerical_grad - dparam)
    denominator = np.maximum(np.abs(numerical_grad) + np.abs(dparam), 1e-8)

    return np.max(numerator/denominator)

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def mse_loss_backward(pred,target):
    return 2 * (pred - target) / pred.size

if(__name__ == "__main__"):
    np.random.seed(0)

    X = np.random.randn(4,3) # batch of 4, 3 input features
    target = np.random.randn(4,2) #  batch of 4, 2 outputs


    layer1 = Layer(3,5)
    relu1 = Relu()
    layer2 = Layer(5,2)

    def full_forward():
        out = layer1.forward(X)
        out = relu1.forward(out)
        out = layer2.forward(out)
        return mse_loss(out, target)

    loss = full_forward()
    dloss = mse_loss_backward(layer2.output, target)
    d = layer2.backward(dloss)
    d = relu1.backward(d)
    d = layer1.backward(d)

    print("layer2.weights:", gradient_check(layer2.weights, layer2.dweights, full_forward))
    print("layer2.biases: ", gradient_check(layer2.biases, layer2.dbiases, full_forward))
    print("layer1.weights:", gradient_check(layer1.weights, layer1.dweights, full_forward))
    print("layer1.biases: ", gradient_check(layer1.biases, layer1.dbiases, full_forward))