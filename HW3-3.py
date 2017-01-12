from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import math
from operator import itemgetter
from collections import Counter


#Loading Iris dataset from scikit-learn
#X features: 0=sepal length, 1=sepal width, 2=petal length, 3=petal width
#y: 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#####################################################################
#First splitting data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Further splitting test data into validation (15%) and test (15%)
X_atest, X_val, y_atest, y_val = train_test_split(X_test, y_test, test_size=0.5)

#####################################################################
#Standardizing features

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_atest)
X_val_std = sc.transform(X_val)

#Reformating train/val/test datasets for convenience
train = np.array(zip(X_train_std,y_train))
val = np.array(zip(X_val_std, y_val))
test = np.array(zip(X_test_std, y_test))

#####################################################################
#k-nearest neighbor classifier
 
#Calculate the euclidean distance between two points
def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))
 
#Use getDistance to calculate all pairwise distances between train and val
def get_neighbours(training_set, val_instance, k):
    distances = [_get_tuple_distance(training_instance, val_instance) for training_instance in training_set]
    # index 1 is the calculated distance between training_instance and val_instance
    sorted_distances = sorted(distances, key=itemgetter(1))
    # extract only training instances
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    # select first k elements
    return sorted_training_instances[:k]
 
def _get_tuple_distance(training_instance, val_instance):
    return (training_instance, get_distance(val_instance, training_instance[0]))
 
#Given an array of nearest neighbours for a val case, tally up their classes to vote on val case class
def get_majority_vote(neighbours):
    # index 1 is the class
    classes = [neighbour[1] for neighbour in neighbours]
    count = Counter(classes)
    return count.most_common()[0][0] 
 
#Setting up main executable method
def main():
    
    #Generate predictions
    predictions = []
 
    #For each instance in the val set, get nearest neighbours and majority vote on predicted class
    for x in range(len(X_val_std)):
            neighbours = get_neighbours(training_set=train, val_instance=val[x][0], k=6)
            majority_vote = get_majority_vote(neighbours)
            predictions.append(majority_vote)
 
    #Performance of the classification
    print '\nAccuracy: ' + str(accuracy_score(y_val, predictions)) + "\n"
    report = classification_report(y_val, predictions, target_names = iris.target_names)
    print 'Classification report: \n\n' + report
 
if __name__ == "__main__":
    main()