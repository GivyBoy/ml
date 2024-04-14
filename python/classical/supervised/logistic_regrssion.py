import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


class logistic_regression:
    def __init__(self, learning_rate=1e-3, iters=10_000, batch_size=32) -> None:
        assert learning_rate > 0, "Learning rate must be greater than 0"
        assert iters > 0, "Number of iterations must be greater than 0"
        self.learning_rate = learning_rate
        self.iters = iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, beta: float = None, model=None) -> None:
        num_samples, num_features = X.shape

        # init weights and bias
        if model and isinstance(model, logistic_regression):
            self.weights = model.weights
            self.bias = model.bias
        else:
            self.weights = np.random.randn(num_features)
            self.bias = 0

        for _ in range(self.iters):
            idx = np.random.randint(0, num_samples, self.batch_size)
            X_batch = np.take(X, idx, axis=0)
            y_batch = np.take(y, idx, axis=0)
            y_pred = self._sigmoid(np.dot(X_batch, self.weights) + self.bias)
            error = y_pred - y_batch
            dw = (2 * (X_batch.T).dot(error)) / num_samples
            db = (2 * np.sum(error)) / num_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


if __name__ == "__main__":

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    model = logistic_regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"accuracy: {logistic_regression.accuracy(y_test, y_pred):.2f}")
