from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import helpers

estimators = 2000
features = 50
# CPU cores for running the RandomForestClassifier.
cpu_cores = 4

print('Start loading data.')
files, labels = helpers.load_chars74k_data()
X, y = helpers.create_dataset(files, labels, with_denoising=True)
print('Data has been loaded.')

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=0.8)

# Normalizing images.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('Start training the model.')
model = RandomForestClassifier(n_estimators=estimators, max_features=features, verbose=True, n_jobs=cpu_cores)
model.fit(x_train, y_train)
print('Model created.')

print('Calculating accuracy.')
pred_test = model.predict(x_test)
print('Test accuracy: ', accuracy_score(y_test, pred_test))
