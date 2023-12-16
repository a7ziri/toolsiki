import numpy as np 

class VanillaLogisticRegression(object):
    '''
    A simple logistic regression for binary classification with gradient descent
    '''

    def __init__(self, learning_rate=0.1, max_iter=100000, tolerance=1e-15):
        # Learning rate for gradient descent
        self._lr = learning_rate

        self._max_iter = max_iter

        # How often to print validation info
        self._validation_freq = 5000

        # Convergence criteria
        self._tolerance = tolerance


    def fit(self, X, y):
        # Add extra dummy feature (x[0] = 1) for bias in linear regression
        X = self.__add_intercept(X)

        n_objects, n_features = X.shape

        # Initialize randomly
        self._weights = np.random.random(n_features)

        # Iterative gradient descent
        for i in range(self._max_iter):
            '''
            Compute logits, gradient, and update weights
            '''
            h = np.dot(X , self._weights)
            z = sigmoid(h)

            grad = np.dot((X/n_objects).T, (z - y))
            self._weights -= self._lr * grad
            # self._weights ...

            if np.linalg.norm(grad) < self._tolerance:
                print("Converged in {} iterations!".format(i))
                break

            if i % self._validation_freq == 0:
                # Compute probabilities
                p = sigmoid(np.dot(X, self._weights))

                # Clip values for numeric stability in logarithm
                p = np.clip(p, 1e-10, 1 - 1e-10)

                # Compute log loss and accuracy
                loss = self.__loss(y, p)
                acc = np.mean((p >= 0.5) == y)

                print("Iteration {}: Loss = {}. Accuracy = {}".format(i, loss, acc))


    def predict(self, X, threshold=0.5):
        X = self.__add_intercept(X)
        proba = self.predict_proba(X)
        predictions = (proba >= threshold).astype(int)
        return predictions
    

    def sigmoid(h):
        return 1/(1+np.exp(-h))




    def predict_proba(self, X):
        # X = self.__add_intercept(X)

        return sigmoid(np.dot(X, self._weights))


    def __add_intercept(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


    def __loss(self, y, p):
      loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
      return loss