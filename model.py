import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#process the data
data_ = pd.read_csv('data.csv')
y = data_.diagnosis
x = data_.drop(['Unnamed: 32','id','diagnosis'],axis = 1 )
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std()) 

#split the data set into train set and test set  
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_n_2, y, test_size = 0.3, random_state=42) 

#high_corr = ['perimeter_mean','radius_mean','compactness_mean','radius_se',
#	'perimeter_se','radius_worst','perimeter_worst','area_worst']
high_corr = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
	'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se',
	'concave points_se','texture_worst','area_worst']
x_train_1 = x_train.drop(high_corr,axis = 1)
x_test_1 = x_test.drop(high_corr,axis = 1)
x1 = data_n_2.drop(high_corr,axis = 1)
fisher = np.zeros(len(x1.columns))
for ind,i in enumerate(x1.columns):
	temp = pd.concat([y,x1[i]],axis=1)
	temp_m = temp[temp.diagnosis=='M']
	temp_b = temp[temp.diagnosis=='B']
	mm = temp_m[i].mean()
	mb = temp_b[i].mean()
	sm = np.mean((temp_m[i]-mm)**2)
	sb = np.mean((temp_b[i]-mb)**2)
	Jw = (mm-mb)**2/(sm+sb)
	fisher[ind] = Jw
features = x1.columns
low_fscore = list(features[np.where(fisher<np.median(fisher))])
x1 = x1.drop(low_fscore,axis=1)
x_train_1 = x_train_1.drop(low_fscore,axis = 1)
x_test_1 = x_test_1.drop(low_fscore,axis = 1)



#SVM with rbf
cmap = sns.cm.rocket_r
clf = svm.SVC(kernel='rbf').fit(x_train_1,y_train)
ac = accuracy_score(y_test,clf.predict(x_test_1))
print "svc.score for train"
print clf.score(x_train_1,y_train)
print "svc.score for test"
print clf.score(x_test_1,y_test)
print('svm accuracy is: ',ac)
#comfusion matrix for SVM with rbf
cm = confusion_matrix(y_test,clf.predict(x_test_1))
ax = sns.heatmap(cm,annot=True,fmt="d",cmap=cmap)
plt.savefig('./picture/svm_rbf.png')
plt.close()


#svm with linear
cmap = sns.cm.rocket_r
clf = svm.SVC(kernel = 'linear').fit(x_train_1,y_train)
ac = accuracy_score(y_test,clf.predict(x_test_1))
print "svc.score for train"
print clf.score(x_train_1,y_train)
print "svc.score for test"
print clf.score(x_test_1,y_test)
cm = confusion_matrix(y_test,clf.predict(x_test_1))
ax = sns.heatmap(cm,annot=True,fmt="d",cmap=cmap)
plt.savefig('./picture/svm_linear.png')
plt.close()



#random forest
cmap = sns.cm.rocket_r
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('on test random forest accuracy is: ',ac)
ac2 = accuracy_score(y_train, clf_rf.predict(x_train))
print('on train random forest accuracy is: ',ac2)
#confusion matrix for random forest
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
ax = sns.heatmap(cm,annot=True,fmt="d",cmap=cmap)
plt.savefig('./picture/RF_predict.png')
plt.close()


#adaBoost classifier

# Using decision stumps due to size of sample.
# Attempting to prevent over-fitting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import RandomizedSearchCV

cmap = sns.cm.rocket_r
stump_clf =  DecisionTreeClassifier(random_state=42, max_depth=1)

param_grid = {
              "base_estimator__max_features": ['auto', 'sqrt', 'log2'],
              "n_estimators": list(range(1,500)),
              "learning_rate": np.linspace(0.01, 1, num=20),
             }
ada_clf = AdaBoostClassifier(base_estimator = stump_clf)
rand_ada = RandomizedSearchCV(ada_clf, param_grid, scoring = 'accuracy', n_iter=100, random_state=42)
rand_ada.fit(x_train_1,y_train)
cm = confusion_matrix(y_test,rand_ada.predict(x_test_1))
ax = sns.heatmap(cm,annot=True,fmt="d",cmap=cmap)
plt.savefig('./picture/adaBoost.png')
plt.close()
# print(rand_ada.best_score_)
# print(rand_ada.best_params_)
# print(rand_ada.best_estimator_)
# print "cv_results_:"
# print(rand_ada.cv_results_)


# RF recursive feature elimination
from sklearn.feature_selection import RFECV
clf_2 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_2, step=1, cv=5,scoring='accuracy') #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
print('Best accuracy :',np.max(rfecv.grid_scores_))




