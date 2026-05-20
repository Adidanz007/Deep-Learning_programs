#5. Build a simple Artificial Neural Network (ANN) with configurable activation and loss functions. Train the network using activation and loss functions.


import numpy as np

class SimpleANN:

    def __init__(self, inp, hid, out, act='sigmoid'):
        self.w1 = np.random.randn(inp, hid)
        self.w2 = np.random.randn(hid, out)
        self.lr = 0.5

        # Activation function selection
        if act == 'sigmoid':
            self.act = lambda x: 1 / (1 + np.exp(-x))
            self.der = lambda x: (s := 1 / (1 + np.exp(-x))) * (1 - s)
        else:   # ReLU
            self.act = lambda x: np.maximum(0, x)
            self.der = lambda x: (x > 0).astype(float)

    def train(self, x, y, epochs=2000):
        for i in range(epochs):
            # -------- Forward Pass --------
            z1 = x @ self.w1
            a1 = self.act(z1)
            z2 = a1 @ self.w2
            output = 1 / (1 + np.exp(-z2))   # Output layer uses sigmoid

            # -------- Loss (MSE) --------
            error = 2 * (output - y) / y.size

            # -------- Backpropagation --------
            dw2 = a1.T @ error
            d_hidden = (error @ self.w2.T) * self.der(z1)
            dw1 = x.T @ d_hidden

            # -------- Update weights --------
            self.w1 -= self.lr * dw1
            self.w2 -= self.lr * dw2
            if i % 500 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Loss: {loss:.4f}")

# -------- Dataset (XOR Problem) --------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# -------- Train ANN --------
model = SimpleANN(2, 4, 1, act='sigmoid')
model.train(X, y)