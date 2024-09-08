import numpy as np

class LogisticRegressions:
  def __init__(self, learning_rate=0.01, iterations=1000):
    self.learning_rate = learning_rate
    self.iterations = iterations

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def compute_loss(self, y, y_pred):
    m = len(y)
    loss = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

  def fit(self, X, y):
    self.m, self.n = X.shape
    self.W = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.y = y

    for i in range(self.iterations):
        self.update_weights()

  def update_weights(self):
    linear_model = np.dot(self.X, self.W) + self.b
    y_pred = self.sigmoid(linear_model)

    # Gradient descent
    dW = (1 / self.m) * np.dot(self.X.T, (y_pred - self.y))
    db = (1 / self.m) * np.sum(y_pred - self.y)

    self.W -= self.learning_rate * dW
    self.b -= self.learning_rate * db

    # Calculate loss
    loss = self.compute_loss(self.y, y_pred)

  def predict(self, X):
    linear_model = np.dot(X, self.W) + self.b
    y_pred = self.sigmoid(linear_model)
    return y_pred >= 0.5
