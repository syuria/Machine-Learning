# load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# train, test split
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


# test options and evaluation matrix
seed = 7
scoring = 'accuracy'


# spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# evaluate each model in turn
results= []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# make predictions on test dataset

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
print("Logistic Regression")
print(accuracy_score(Y_test, predictions))
print( confusion_matrix(Y_test, predictions))
print( classification_report(Y_test, predictions))


# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_test)
print( "Linear Discriminant Analysis")
print( accuracy_score(Y_test, predictions))
print( confusion_matrix(Y_test, predictions))
print( classification_report(Y_test, predictions))


# KNeighbours Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print( "KNeighbours Classifier")
print( accuracy_score(Y_test, predictions))
print( confusion_matrix(Y_test, predictions))
print( classification_report(Y_test, predictions))


# Decision Tree Classifier
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_test)
print( "Decision Tree Classifier")
print( accuracy_score(Y_test, predictions))
print( confusion_matrix(Y_test, predictions))
print( classification_report(Y_test, predictions))


# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_test)
print( "Gaussian Naive Bayes")
print( accuracy_score(Y_test, predictions))
print( confusion_matrix(Y_test, predictions))
print( classification_report(Y_test, predictions))


# SVM
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)
print( "Support Vector Machines")
print( accuracy_score(Y_test, predictions))
print( confusion_matrix(Y_test, predictions))
print( classification_report(Y_test, predictions))