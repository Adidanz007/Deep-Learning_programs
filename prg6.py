#6. Implement the categorical cross-entropy loss function with both forward and backward passes and verify loss computation


import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cce_forward(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cce_backward(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]


y_true = np.array([
    [1,0,0],
    [0,1,0]
])

logits = np.array([
    [2.0,1.0,0.1],
    [1.0,3.0,0.2]
])

probs = softmax(logits)
loss = cce_forward(y_true, probs)
grad = cce_backward(y_true, probs)

print("Probabilities:")
print(probs)

print("\nLoss:")
print(loss)

print("\nGradient:")
print(grad)