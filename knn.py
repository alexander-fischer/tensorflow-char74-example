import numpy as np

from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import helpers
import image_detection as detector

print('Start loading data.')
files, labels = helpers.load_chars74k_data()
X, y = helpers.create_dataset(files, labels)
print('Data has been loaded.')

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)

print('Start training the model.')
model = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
model.fit(x_train, y_train)
print('Model created.')

print('Calculating accuracy.')
pred_test = model.predict(x_test)
print('Test accuracy: ', accuracy_score(y_test, pred_test))