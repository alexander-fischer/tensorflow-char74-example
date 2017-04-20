from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
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
model = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=4)
model.fit(x_train, y_train)
print('Model created.')

print('Calculating accuracy.')
pred_test = model.predict(x_test)
print('Test accuracy: ', accuracy_score(y_test, pred_test))
