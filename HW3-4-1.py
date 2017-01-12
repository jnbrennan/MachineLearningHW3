from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


#Loading Iris dataset from scikit-learn
#X features: 0=sepal length, 1=sepal width, 2=petal length, 3=petal width
#y: 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#####################################################################
#First splitting data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Further splitting test data into validation/development (15%) and test (15%)
X_atest, X_val, y_atest, y_val = train_test_split(X_test, y_test, test_size=0.5)

#####################################################################
#Standardizing features

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_atest)
X_val_std = sc.transform(X_val)

#####################################################################
#Train logistic regression moel with standard classifier parameters

lr = LogisticRegression(C=10.0, random_state=None)
lr.fit(X_train_std, y_train)
print(lr)

#####################################################################
#Making predictions

y_pred = lr.predict(X_test_std)

#####################################################################
#Checking accuracy and f1-score

print('Misclassified samples: %d' % (y_atest != y_pred).sum())
print('Accuracy:')
print accuracy_score(y_atest, y_pred)
print(metrics.classification_report(y_atest, y_pred))