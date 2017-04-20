from collections import Counter

from sklearn.ensemble import RandomForestClassifier
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
model = RandomForestClassifier(n_estimators=2000, max_features=50, verbose=True, n_jobs=4)
model.fit(x_train, y_train)
print('Model created.')

print('Calculating accuracy.')
pred_test = model.predict(x_test)
print('Test accuracy: ', accuracy_score(y_test, pred_test))

detection1 = './detection-images/detection-1.jpg'
samples1 = detector.sliding_window(detection1)

print('Start detection on example image: ', detection1)
predictions = model.predict(samples1)
value_list = []

for pred in predictions:
    value_list.append(helpers.num_to_char(pred))

print('Predicted values on image:')
print(Counter(value_list))
