# https://en.wikipedia.org/wiki/Cohen%27s_kappa
# 重み付きk係数とかともいう

from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np

def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            # Weight
            wij = ((i - j) ** 2.0)
            # Observed
            oij = c_matrix[i, j]
            # Expected
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij
    return 1.0 - numer / denom

# Example
if __name__ == '__main__':
    y_true = [1, 2, 3, 4, 3]
    y_pred = [2, 2, 4, 4, 5]
    c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    print(quadratic_weighted_kappa(c_matrix))
    # Also just using sklearn
    print(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
