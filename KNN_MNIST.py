from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

X = X.astype(np.float64)
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
knn = KNeighborsClassifier()

param_grid=[
    { 'n_neighbors':[2,3,4], 'weights' :['uniform', 'distance']}
]

grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train_5)
print(grid_search.best_params_)

print(cross_val_score(knn, X_train, y_train_5, cv=3, scoring='accuracy'))
