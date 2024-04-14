import numpy as np
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")


class BaseRegression(ABC):  # Parent Class
    def __init__(self, learning_rate: float = 1e-3, iters: int = 30_000, batch_size: int = 32) -> None:
        self.learning_rate = learning_rate
        self.iters = iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _approx(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray, model=None) -> None:
        num_samples, num_features = X.shape

        # init weights and bias
        if model and isinstance(model, BaseRegression):
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
            y_pred = self._approx(X_batch)
            error = y_pred - y_batch
            dw = (2 * (X_batch.T).dot(error)) / num_samples
            db = (2 * np.sum(error)) / num_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


class linear_regression(BaseRegression):  # Child Class
    def _approx(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum((y_true - y_pred) ** 2) / len(y_true)


class logistic_regression(BaseRegression):  # Child Class
    def _approx(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X.dot(self.weights) + self.bias)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self._sigmoid(X.dot(self.weights) + self.bias) > 0.5, 1, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    regression = linear_regression()
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=17)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    regressor = linear_regression(learning_rate=1e-3, iters=30_000)
    regressor.fit(X_train, y_train)
    regressor_pred = regressor.predict(X_test)

    regressor_mse = linear_regression.mse(y_test, regressor_pred)

    print(f"Regressor MSE: {regressor_mse}")

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    model = logistic_regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"accuracy: {logistic_regression.accuracy(y_test, y_pred):.2f}")
