# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


# In[2]:

adultData=pd.read_csv("adult.data", header=None, na_values=[' ?'])
adultData.columns=["age", "workclass", "fnlwgt", "education","education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "Target"]


# In[3]:

adultData['Target'].replace([' <=50K',' >50K'],[0,1],inplace=True)


# In[4]:

categoricalColumns=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']


# In[5]:

#filling missing values in categorical columns with mode
categoricalNAColumns=['workclass','occupation','native-country']
for columns in categoricalNAColumns:
    adultData[columns].fillna((mode(adultData[columns]).mode[0]), inplace=True)


# In[6]:

#Encodes categorical features into numerical features using dummy encoding
for columns in categoricalColumns:
    adultData = adultData.join(pd.get_dummies(adultData[columns], prefix=columns))
    del adultData[columns]
encodedAdultData=adultData


# In[7]:

numericalColumns=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']


# In[8]:

for columns in numericalColumns:
    encodedAdultData[columns].fillna((encodedAdultData[columns].mean()),inplace=True)


# In[9]:

#Transforms numerical data from existing scale to specified scale,Data transformation is required since numerical features can take various ranges and a common range is required among all the numerical features
scaler = MinMaxScaler()
for columns in numericalColumns:
    encodedAdultData[[columns]]=scaler.fit_transform(encodedAdultData[[columns]])
transformedAdultData=encodedAdultData


# In[10]:

#splitting data in training data and test data
msk = np.random.rand(len(transformedAdultData)) < 0.7
trainData = transformedAdultData[msk]
testData = transformedAdultData[~msk]
targetTrain = trainData['Target']
del trainData['Target']
targetTest = testData['Target']
del testData['Target']


# In[ ]:

#Applying K nearest neighbors algorithm on the dataset
k = []
kROC = []
for n in range(5,20):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn, trainData, targetTrain, cv=10, scoring='roc_auc')
    k.append(n)
    kROC.append(scores.mean())
    print "neighbors: ", n, "score: ", scores.mean()


# In[42]:

#plotting ROC curve against K for K nearest neighbors
fig, ax = plt.subplots()
ax.set_title('Area under curve Vs K')
ax.set_xlim(min(k)-1, max(k)+1)
ax.legend(loc='lower right')
plt.xlabel("K")
plt.ylabel("ROC-AUC-Score")
plt.plot(k,kROC, color ='g', lw=2)
plt.savefig('KNN - Area under curve Vs K')
plt.close()


# In[45]:

knnBestK = KNeighborsClassifier(n_neighbors=25)
knnBestK.fit(trainData, targetTrain)
# make predictions
expectedK = targetTest
predictedK = knnBestK.predict(testData)
# summarize the fit of the model
print "roc_auc_score", roc_auc_score(expectedK, predictedK)
#Plotting ROC curve for the best k after tuning the parameters
false_positive_rate, true_positive_rate, thresholds = roc_curve(expectedK, predictedK)
roc_auc = auc(false_positive_rate, true_positive_rate)
fig, ax = plt.subplots()
ax.set_title('K nearest neighbors(Best K): Area under curve')
ax.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
ax.legend(loc='lower right')
ax.plot([0,1],[0,1],'r--')
ax.set_xlim([-0.1,1.2])
ax.set_ylim([-0.1,1.2])
ax.set_xlabel('True Positive Rate')
ax.set_ylabel('False Positive Rate')
plt.savefig('K nearest neighbors(Best K) - Area under curve')
plt.close()


# In[19]:

#Applies decision tree algorithm on the dataset, by tuning various parameters
decisionTreeDepths = []
decisionTreeROC = []
for depth in range(3,50):
    dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth)
    scores = cross_val_score(dt, trainData, targetTrain, cv=10, scoring='roc_auc')
    decisionTreeDepths.append(depth)
    decisionTreeROC.append(scores.mean())
    print "Depth: ", depth, "score: ", scores.mean()


# In[46]:

#plotting ROC curve against depth for decision tree
fig, ax = plt.subplots()
ax.set_title('Area under curve Vs Depth')
ax.set_xlim(min(decisionTreeDepths)-1, max(decisionTreeDepths)+1)
ax.legend(loc='lower right')
plt.xlabel("Depth")
plt.ylabel("ROC-AUC-Score")
plt.plot(decisionTreeDepths,decisionTreeROC, color ='g', lw=2)
plt.savefig('DT - Area under curve Vs Depth')
plt.close()


# In[47]:

dtBestDepth = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=8)
dtBestDepth.fit(trainData, targetTrain)
# make predictions
expectedDT = targetTest
predictedDT = dtBestDepth.predict(testData)
# summarize the fit of the model
print "roc_auc_score", roc_auc_score(expectedDT, predictedDT)
#Plotting ROC curve for the best depth after tuning the parameters
false_positive_rate, true_positive_rate, thresholds = roc_curve(expectedDT, predictedDT)
roc_auc = auc(false_positive_rate, true_positive_rate)
fig, ax = plt.subplots()
ax.set_title('Decision Tree(Best Depth): Area under curve')
ax.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
ax.legend(loc='lower right')
ax.plot([0,1],[0,1],'r--')
ax.set_xlim([-0.1,1.2])
ax.set_ylim([-0.1,1.2])
ax.set_xlabel('True Positive Rate')
ax.set_ylabel('False Positive Rate')
plt.savefig('Decision Tree(Best Depth) - Area under curve')
plt.close()


# In[65]:

# Applies random forest algorithm on the dataset, by tuning various parameters
randomForestDepths = []
randomForestNEstimators = []
randomForestROC = []
#for depth in range(3,50):
for n in range(10,110,10):
    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=14, n_estimators=n)
    scores = cross_val_score(rfc, trainData, targetTrain, cv=10, scoring='roc_auc')
    #randomForestDepths.append(depth)
    randomForestNEstimators.append(n)
    randomForestROC.append(scores.mean())
    #print "Depth: ", depth, "score: ", scores.mean()
    print "n_estimators: ", n, "score: ", scores.mean()


# In[66]:

#plotting ROC curve against depth for random forest
fig, ax = plt.subplots()
ax.set_title('Area under curve Vs Number of trees')
ax.set_xlim(min(randomForestNEstimators)-1, max(randomForestNEstimators)+1)
ax.legend(loc='lower right')
plt.xlabel("Number of trees")
plt.ylabel("ROC-AUC-Score")
plt.plot(randomForestNEstimators,randomForestROC, color ='g', lw=2)
plt.savefig('RF - Area under curve Vs Number of trees')
plt.close()


# In[68]:

rfcBestDepth = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=14, n_estimators=80)
rfcBestDepth.fit(trainData, targetTrain)
# make predictions
expectedRF = targetTest
predictedRF = rfcBestDepth.predict(testData)
# summarize the fit of the model
print "roc_auc_score", roc_auc_score(expectedRF, predictedRF)
#Plotting ROC curve for the best depth and the number of trees after tuning the parameters
false_positive_rate, true_positive_rate, thresholds = roc_curve(expectedRF, predictedRF)
roc_auc = auc(false_positive_rate, true_positive_rate)
fig, ax = plt.subplots()
ax.set_title('Random Forest(Best number of trees): Area under curve')
ax.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
ax.legend(loc='lower right')
ax.plot([0,1],[0,1],'r--')
ax.set_xlim([-0.1,1.2])
ax.set_ylim([-0.1,1.2])
ax.set_xlabel('True Positive Rate')
ax.set_ylabel('False Positive Rate')
plt.savefig('Random Forest(Best number of trees) - Area under curve')
plt.close()


# In[30]:

#Applies support vector machine algorithm on the dataset, by tuning various parameters
svmC = [0.01, 0.1, 1, 10]
svmROC = []
for c in svmC:
    svm = SVC(C=c, cache_size=200, kernel='linear')
    scores = cross_val_score(svm, trainData, targetTrain, cv=10, scoring='roc_auc')
    svmROC.append(scores.mean())
    print "C: ", c, "score: ", scores.mean()


# In[50]:

#plotting ROC curve against C for SVM
fig, ax = plt.subplots()
ax.set_title('Area under curve Vs C')
ax.set_xlim(min(svmC)-1, max(svmC)+1)
ax.legend(loc='lower right')
plt.xlabel("C")
plt.ylabel("ROC-AUC-Score")
plt.plot(svmC,svmROC, color ='g', lw=2)
plt.savefig('SVM - Area under curve Vs C')
plt.close()


# In[51]:

svmBestC = SVC(C=10, cache_size=200, kernel='linear')
svmBestC.fit(trainData,targetTrain)
# make predictions
expectedSVM = targetTest
predictedSVM = svmBestC.predict(testData)
# summarize the fit of the model
print "roc_auc_score", roc_auc_score(expectedSVM, predictedSVM)
#Plotting ROC curve for the best C after tuning the parameters
false_positive_rate, true_positive_rate, thresholds = roc_curve(expectedSVM, predictedSVM)
roc_auc = auc(false_positive_rate, true_positive_rate)
fig, ax = plt.subplots()
ax.set_title('SVM(Best C): Area under curve')
ax.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
ax.legend(loc='lower right')
ax.plot([0,1],[0,1],'r--')
ax.set_xlim([-0.1,1.2])
ax.set_ylim([-0.1,1.2])
ax.set_xlabel('True Positive Rate')
ax.set_ylabel('False Positive Rate')
plt.savefig('SVM(Best C) - Area under curve')
plt.close()


# In[56]:

adaboostBaseEstimator = DecisionTreeClassifier(criterion='gini', max_depth=5,min_samples_leaf=5)
adaNumEstimators = []
adaROC = []
for estimators in range (10,100,10):
    adaclf = AdaBoostClassifier(base_estimator = adaboostBaseEstimator, n_estimators=estimators)
    scores = cross_val_score(adaclf, trainData, targetTrain, cv=10, scoring='roc_auc')
    adaNumEstimators.append(estimators)
    adaROC.append(scores.mean())
    print "Num estimators: ", estimators, "ROC-AUC-Score: ", scores.mean()


# In[58]:

#plotting ROC curve against Number of Trees for adaboost
fig, ax = plt.subplots()
ax.set_title('Area under curve Vs Number of Trees')
ax.set_xlim(min(adaNumEstimators)-1, max(adaNumEstimators)+1)
plt.xlabel("Number of Trees")
plt.ylabel("ROC-AUC-Score")
plt.plot(adaNumEstimators,adaROC, color ='g', lw=2)
plt.savefig('Adaboost - Area under curve Vs Number of Trees')
plt.close()


# In[61]:

baseDecisionTree = DecisionTreeClassifier(criterion='gini', max_depth=5,min_samples_leaf=5)
bestAdaboost = AdaBoostClassifier(base_estimator = baseDecisionTree, n_estimators=10)
bestAdaboost.fit(trainData, targetTrain)
expectedAB = targetTest
predictedAB = bestAdaboost.predict(testData)
print "roc_auc_score", roc_auc_score(expectedAB, predictedAB)
#Plotting ROC curve for the best number of trees after tuning the parameters
false_positive_rate, true_positive_rate, thresholds = roc_curve(expectedAB, predictedAB)
roc_auc = auc(false_positive_rate, true_positive_rate)
fig, ax = plt.subplots()
ax.set_title('Adaboost(Best tree): Area under curve')
ax.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
ax.legend(loc='lower right')
ax.plot([0,1],[0,1],'r--')
ax.set_xlim([-0.1,1.2])
ax.set_ylim([-0.1,1.2])
ax.set_xlabel('True Positive Rate')
ax.set_ylabel('False Positive Rate')
plt.savefig('Adaboost(Best tree) - Area under curve')
plt.close()