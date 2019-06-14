# Load libraries
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
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
import pickle

def load_data(fp='data_30bp_72.txt'):
	Y = []
	with open(fp, 'r') as f:
		for line in f:
			seq, resp = line.strip().split()
			if resp=='NA':
				resp = -10.
			#Y.append(float(resp))
			if float(resp)>0:
				Y.append(0)
			else:
				Y.append(1)
	Y = np.array(Y)
	return Y


X = pd.read_csv("feature_used.csv")
Y = load_data()

#remove the NA rows in the X feature matrix
inds = pd.isnull(X).any(1).nonzero()[0]
X_new = X.drop(inds)
Y_new = np.delete(Y,(inds),axis=0)

X_train, tmp_X, Y_train, tmp_y = train_test_split(
	X_new, Y_new, test_size = 0.2, random_state=111)
X_val, X_test, Y_val, Y_test = train_test_split(
	tmp_X, tmp_y, test_size=0.5, random_state=222)


model = SVC()
model.fit(X_train, Y_train)

def write_pred(y_true, y_pred):
	with open('log.txt', 'w') as fo:
		auroc = metrics.roc_auc_score(y_true, y_pred)
		aupr = metrics.average_precision_score(y_true, y_pred)
		fo.write("# auroc=%.4f\n"%auroc)
		fo.write("# aupr=%.4f\n"%aupr)
		for i in range(len(y_true)):
			fo.write("{}\t{}\n".format(y_true[i], y_pred[i]))


y_pred = model.predict(X_test)
write_pred(Y_test, y_pred)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
# Test options and evaluation metric
seed = 166	
scoring = 'accuracy'

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()
plt.savefig('comparison.png')

filename = 'finalized_5930_models.sav'
pickle.dump(models, open(filename, 'wb'))

filename = 'SVC.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open('finalized_5930_models.sav', 'rb'))
result = model.score(X_test, Y_test)
print(result)

#figure out the weighted for each model

