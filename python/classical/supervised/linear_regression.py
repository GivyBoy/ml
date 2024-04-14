import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets


class linear_regression:
    def __init__(self, learning_rate=1e-3, iters=30_000, batch_size=32) -> None:
        assert learning_rate > 0, "Learning rate must be greater than 0"
        assert iters > 0, "Number of iterations must be greater than 0"
        self.learning_rate = learning_rate
        self.iters = iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, model=None) -> None:
        num_samples, num_features = X.shape

        # init weights and bias
        if model and isinstance(model, linear_regression):
            self.weights = model.weights
            self.bias = model.bias
        else:
            self.weights = np.random.randn(num_features)
            self.bias = 0

        # SGD algo
        for _ in range(self.iters):
            idx = np.random.randint(0, num_samples, self.batch_size)
            X_batch = np.take(X, idx, axis=0)
            y_batch = np.take(y, idx, axis=0)
            y_pred = self.predict(X_batch)
            error = y_pred - y_batch
            dw = (2 * (X_batch.T).dot(error)) / num_samples
            db = (2 * np.sum(error)) / num_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum((y_true - y_pred) ** 2) / len(y_true)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


if __name__ == "__main__":

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=17)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    lin_reg = LinearRegression()
    regressor = linear_regression(learning_rate=1e-3, iters=30_000)
    reg = linear_regression(learning_rate=1e-3, iters=30_000)
    lin_reg.fit(X_train, y_train)
    regressor.fit(X_train, y_train)
    reg.fit(X_train, y_train, regressor)
    lin_reg_pred = lin_reg.predict(X_test)
    regressor_pred = regressor.predict(X_test)

    lin_reg_mse = linear_regression.mse(y_test, lin_reg_pred)
    regressor_mse = linear_regression.mse(y_test, regressor_pred)
    reg_mse = linear_regression.mse(y_test, reg.predict(X_test))

    print(f"Linear Regression MSE: {lin_reg_mse} | Regressor MSE: {regressor_mse} | Reg MSE: {reg_mse}")
    print(math.isclose(reg_mse, regressor_mse, rel_tol=0.02))
