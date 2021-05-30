import pandas as pd
import numpy as np

df = pd.read_csv("sonar.csv")

null = df.isnull().sum()

X = df.drop(["Class"], axis=1)
y = df["Class"]

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

y= label_encoder.fit_transform(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 3)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


CVAccuracy=[]
for j in range(1,60):
    knn = KNeighborsClassifier(n_neighbors = j)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    CVAccuracy.append([scores.mean(),j])
KNN_dff = pd.DataFrame (CVAccuracy,columns=['Validation Accuracy','NeighbourSize'])


# k value = 3

knn =KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
y_predknn=knn.predict(X_test)
print('Standard KNN Accuracy:', accuracy_score(y_test, y_predknn))


#wrapper
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

KN_SFSAccuracy=[]
for j in range(1,60):
    selector = SFS(LogisticRegression(max_iter=1000), k_features=j)
    X_train_new = selector.fit_transform(X_train,y_train)
    X_test_new = selector.transform(X_test)

    knn.fit(X_train_new,y_train)
    ypred_train = knn.predict(X_train_new)
    ypred_test = knn.predict(X_test_new)
    
    score = accuracy_score(y_train,ypred_train)

    KN_SFSAccuracy.append([score,j])
KNN_wrapper_dff = pd.DataFrame (KN_SFSAccuracy,columns=['Accuracy','n features'])


#n_features = 41,42,43

selector = SFS(LogisticRegression(max_iter=1000), k_features=43)
X_train_new = selector.fit_transform(X_train,y_train)
X_test_new = selector.transform(X_test)

knn.fit(X_train_new,y_train)
ypred_train = knn.predict(X_train_new)
ypred_test = knn.predict(X_test_new)
print("KNN with Wrapper Accuracy:",accuracy_score(y_test,ypred_test))

#filter
from sklearn.feature_selection import VarianceThreshold
import collections

varModel=VarianceThreshold(threshold=0)
varModel.fit(X_train)
constArr=varModel.get_support()
collections.Counter(constArr)


correlated_features = set()
correlation_matrix = X_train.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
knn_X_train = X_train
knn_X_test = X_test
knn_X_train.drop(labels=correlated_features, axis=1, inplace=True)
knn_X_test.drop(labels=correlated_features, axis=1, inplace=True)

knn.fit(knn_X_train,y_train)
ypred_train = knn.predict(knn_X_train)
ypred_test = knn.predict(knn_X_test)
print("KNN with Filter Accuracy:",accuracy_score(y_test,ypred_test))




#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#standard
logreg = LogisticRegression(solver="liblinear")
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
print('Standard LogisticRegression Accuracy:', accuracy_score(y_test, logreg_pred))


#embedded

lr=LogisticRegression(solver='liblinear')
C_param_range = [0.001,0.01,0.1,1,10,100,1000,10000]
penalties=['l1','l2']
# create grid
params = {
 'C': C_param_range,
 'penalty': penalties,
 }

lr_grid = GridSearchCV(estimator = lr, param_grid = params,cv = 5, verbose=2, scoring='accuracy',n_jobs = -1)

lr_grid.fit(X_train, y_train)

print(lr_grid.best_params_)
# {'C': 100, 'penalty': 'l2'}

logreg=LogisticRegression(C=100,penalty='l2',solver='liblinear')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print('Logistic Regression with Embedded Accuracy:', accuracy_score(y_test, y_pred))

#wrapper
LR_FSAccuracy=[]
for j in range(1,60):
    selector = SFS(LogisticRegression(max_iter=1000), k_features=j)
    X_train_new = selector.fit_transform(X_train,y_train)
    X_test_new = selector.transform(X_test)

    logreg.fit(X_train_new,y_train)
    ypred_train = logreg.predict(X_train_new)
    ypred_test = logreg.predict(X_test_new)
    
    score = accuracy_score(y_test,ypred_test)
    LR_FSAccuracy.append([score,j])
LR_wrapper_dff = pd.DataFrame (LR_FSAccuracy,columns=['Accuracy','n features'])


# n_features : 3


selector = SFS(LogisticRegression(max_iter=1000), k_features=3)
X_train_new = selector.fit_transform(X_train,y_train)
X_test_new = selector.transform(X_test)

logreg.fit(X_train_new,y_train)
ypred_train = logreg.predict(X_train_new)
ypred_test = logreg.predict(X_test_new)
print("Logistic Regression with Wrapper Accuracy:",accuracy_score(y_test,ypred_test))


#filter
from sklearn.feature_selection import VarianceThreshold
import collections

varModel=VarianceThreshold(threshold=0)
varModel.fit(X_train)
constArr=varModel.get_support()
collections.Counter(constArr)


correlated_features = set()
correlation_matrix = X_train.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
lr_X_train = X_train
lr_X_test = X_test
lr_X_train.drop(labels=correlated_features, axis=1, inplace=True)
lr_X_test.drop(labels=correlated_features, axis=1, inplace=True)

logreg.fit(lr_X_train,y_train)
ypred_train = logreg.predict(lr_X_train)
ypred_test = logreg.predict(le_X_test)
print("Logistic Regression with Filter Accuracy:",accuracy_score(y_test,ypred_test))


#Random Forest

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Standard Random Forest Accuracy:",accuracy_score(y_test,y_pred))

#wrapper
RF_SFSAccuracy=[]
for j in range(1,60):
    selector = SFS(LogisticRegression(max_iter=1000), k_features=j)
    X_train_new = selector.fit_transform(X_train,y_train)
    X_test_new = selector.transform(X_test)

    clf.fit(X_train_new,y_train)
    ypred_train = clf.predict(X_train_new)
    ypred_test = clf.predict(X_test_new)
    
    score = accuracy_score(y_test,ypred_test)
    RF_SFSAccuracy.append([score,j])
RF_wrapper_dff = pd.DataFrame (RF_SFSAccuracy,columns=['Accuracy','n features'])


# n_features : 57

from sklearn.feature_selection import SFS

selector = SFS(LogisticRegression(max_iter=1000), k_features=57)
X_train_new = selector.fit_transform(X_train,y_train)
X_test_new = selector.transform(X_test)

clf.fit(X_train_new,y_train)
ypred_train = clf.predict(X_train_new)
ypred_test = clf.predict(X_test_new)
print("Random Forest with Wrapper Accuracy:",accuracy_score(y_test,ypred_test))

#filter
from sklearn.feature_selection import VarianceThreshold
import collections

varModel=VarianceThreshold(threshold=0)
varModel.fit(X_train)
constArr=varModel.get_support()
collections.Counter(constArr)


correlated_features = set()
correlation_matrix = X_train.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
clf_X_train = X_train
clf_X_test = X_test
clf_X_train.drop(labels=correlated_features, axis=1, inplace=True)
clf_X_test.drop(labels=correlated_features, axis=1, inplace=True)

clf.fit(clf_X_train,y_train)
ypred_train = clf.predict(clf_X_train)
ypred_test = clf.predict(clf_X_test)
print("Random Forest with Filter Accuracy:",accuracy_score(y_test,ypred_test))

#SVM
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Standard SVM Accuracy:",accuracy_score(y_test,ypred_test))

#wrapper
SVM_SFSAccuracy=[]
for j in range(1,60):
    selector = SFS(LogisticRegression(max_iter=1000), k_features=j)
    X_train_new = selector.fit_transform(X_train,y_train)
    X_test_new = selector.transform(X_test)

    svm.fit(X_train_new,y_train)
    ypred_train = svm.predict(X_train_new)
    ypred_test = svm.predict(X_test_new)
    
    score = accuracy_score(y_train,ypred_train)
    SVM_SFSAccuracy.append([score,j])
SVM_wrapper_dff = pd.DataFrame (SVM_SFSAccuracy,columns=['Accuracy','n features'])


# n_features : 38,43

from sklearn.feature_selection import SFS

selector = SFS(LogisticRegression(max_iter=1000), k_features=43)
X_train_new = selector.fit_transform(X_train,y_train)
X_test_new = selector.transform(X_test)

svm.fit(X_train_new,y_train)
ypred_train = svm.predict(X_train_new)
ypred_test = svm.predict(X_test_new)
print("SVM with Wrapper Accuracy:",accuracy_score(y_test,ypred_test))


#filter
from sklearn.feature_selection import VarianceThreshold
import collections

varModel=VarianceThreshold(threshold=0)
varModel.fit(X_train)
constArr=varModel.get_support()
collections.Counter(constArr)


correlated_features = set()
correlation_matrix = X_train.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
svm_X_train = X_train
svm_X_test = X_test
svm_X_train.drop(labels=correlated_features, axis=1, inplace=True)
svm_X_test.drop(labels=correlated_features, axis=1, inplace=True)

svm.fit(clf_X_train,y_train)
ypred_train = svm.predict(clf_X_train)
ypred_test = svm.predict(clf_X_test)
print("SVM with Filter Accuracy:",accuracy_score(y_test,ypred_test))
