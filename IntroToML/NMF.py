# Non-negative Matrix Factorizationの実装

from sklearn.decomposition import NMF

model = NMF(n_components=5, init="random", random_state=71)
model.fit(train_x)

train_x = model.transform(train_x)
test_x = model.transform(test_x)