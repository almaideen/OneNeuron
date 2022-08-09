import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,'running_logs.log'),level=logging.INFO,format=logging_str)

class perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4  # small weights initialization
        logging.info(f"Initial weights: \n {self.weights}")
        self.eta = eta  # Learning Rate
        self.epochs = epochs  # Iterations

    def activationfunction(self, inputs, weights):
        z = np.dot(inputs, weights)  # z = w1.x1+w2.x2+w0.x0
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        self.X = X
        self.y = y

        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]  # concatinate x with bias
        logging.info(f"X with Bias: \n {X_with_bias}")

        for epoch in tqdm(range(self.epochs),total=self.epochs,desc="Training the Model"):
            logging.info("--" * 10)
            logging.info(f"for epoch: \n {epoch}")
            logging.info("--" * 10)

            y_hat = self.activationfunction(X_with_bias, self.weights)  # forward propagation
            logging.info(f"predicted value after forward propagation:\n {y_hat}")
            self.error = self.y - y_hat
            logging.info(f"Error: \n {self.error}")
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)  # backward propagation
            logging.info(f"updated weight after {epoch}/{self.epochs} : {self.weights}")
            logging.info("####" * 10)

    def predict(self, X):
        X_with_bias = np.c_[X, -np.ones((len(X), 1))]
        return self.activationfunction(X_with_bias, self.weights)

    def totalloss(self):
        total_loss = np.sum(self.error)
        logging.info(f"Total Loss: {total_loss}")
        return total_loss