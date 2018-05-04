from sklearn import svm
import json

def read_json(filename):
    file = open(filename, 'r')
    matrix = json.load(file)
    file.close()
    return matrix


x_name = 'traditional-training.json'
y_name = ''
xtest_name = 'traditional-testing.json'
ytest_name = 't'

x = read_json(x_name)
y = [0] * 120 + [1] * 120
xtest = read_json(xtest_name)
ytest = [0] * 31 + [1] * 31

clf = svm.SVC()
clf.fit(x, y)
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

predictions = clf.predict(xtest)
difference = predictions - ytest

tp = 0
tn = 0
fp = 0
fn = 0

for pair in zip(predictions, ytest):
    if pair == (0, 0):
        tn += 1
    if pair == (0, 1):
        fn += 1
    if pair == (1, 0):
        fp += 1
    if pair == (1, 1):
        tp += 1

fpr = fp / (tn + fp)
tpr = tp / (tp + fn)

print((fpr, tpr))
