import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
DIABETES = pd.read_csv('DIABETES.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DIABETES.loc[:,DIABETES.columns != 'Outcome'], DIABETES['Outcome'], stratify=DIABETES['Outcome'], random_state=0)


#Gradient Boost 92.4% and 80%

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
#print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))
#print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))
pickle.dump(gb, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

'''
#Neural Netwroks Model
print('\n Neural Network')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=25)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
pickle.dump(mlp, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))




#Decision tree accuracy: 90% and 70%
print('\n Decision Tree')
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))



#Random Forest accuracy 95% and 78%
print('\n Random Forest Model')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=33)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))



#Gradient Boost 92.4% and 80%
print('\n GRADIENT BOOST Model')
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))
pickle.dump(gb, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))



#SVM 76% and 78%
print('\n SVM Model')
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test, y_test)))



#KNN 77% and 77%
print('\n KNN Model')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.3f}'.format(knn.score(X_test, y_test)))

Accuracy Result:    


 Neural Network
Accuracy on training set: 0.790
Accuracy on test set: 0.755

 Decision Tree
Accuracy on training set: 1.000
Accuracy on test set: 0.760

 Random Forest Model
Accuracy on training set: 1.000
Accuracy on test set: 0.776

 GRADIENT BOOST Model
Accuracy on training set: 0.922
Accuracy on test set: 0.797

 SVM Model
Accuracy on training set: 0.771
Accuracy on test set: 0.766

 KNN Model
Accuracy of K-NN classifier on training set: 0.771
Accuracy of K-NN classifier on test set: 0.776
    
'''
