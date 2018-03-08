from sklearn import svm
xtest = []
ytest = []
x = []
y = []
for filename in sys.argv:
    if "test" in filename:
        xtest.append(np.fromfile(filename, dtype = np.dtype("i4")) / 10000)
        if "positive" in filename:
            ytest.append(1)
        else:
            ytest.append(0)
    if "training" in filename:
        x.append(np.fromfile(filename, dtype = np.dtype("i4")) / 10000)
        if "positive" in filename:
            y.append(1)
        else:
            y.append(0)

clf = svm.SVC()
clf.fit(x, y)  
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

predictions = clf.predict(xtest)
difference = predictions - ytest
error = 0.5*((difference)**2)
epochError = sum(error)/len(error)
