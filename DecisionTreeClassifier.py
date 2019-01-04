'''
Created on Dec 8, 2018

@author: adarsh
'''

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold


#Feature engineering

def mapStringToInt( dataFrameCopy):
    '''Creating a dictionary for the attributes to map from String to int'''
    convertBuyToString = {'buying':{'vhigh': 1,'high':2,'med':3,'low':4}}
    dataFrameCopy.replace(convertBuyToString,inplace=True)

    convertMaintToString = {'maint':{'vhigh': 1,'high':2,'med':3,'low':4}}
    dataFrameCopy.replace(convertMaintToString,inplace=True)

    convertSafetyToString = {'safety':{'high':2,'med':3,'low':4}}
    dataFrameCopy.replace(convertSafetyToString,inplace=True)

    convertLugBootToString = {'lug_boot':{'big':2,'med':3,'small':4}}
    dataFrameCopy.replace(convertLugBootToString,inplace=True)

    convertPersonsToString = {'persons':{'more':5}}
    dataFrameCopy.replace(convertPersonsToString,inplace=True)

    convertDoorsToString = {'doors':{'5more':6}}
    dataFrameCopy.replace(convertDoorsToString,inplace=True)
        
    return dataFrameCopy

def predictCLSingleTrainTest(classifierBuildT,features_train,classLabel_train,features_test,classLabel_test):
    '''Select the classifier and fit the data to build the model'''
    
    classifierDT = classifierBuildT.fit(features_train,classLabel_train)
 
    '''Save the built model to a file and re-load from file(serialization and de-serialization)'''
    saveTheModel = 'decisionTree_model.sav'
    pickle.dump(classifierDT, open(saveTheModel, 'wb'))
    # load the model from disk
    loadedDT_model = pickle.load(open(saveTheModel, 'rb'))
 
    '''Use the fit model to predict the class on test data '''
    pred_classLabel = loadedDT_model.predict(features_test)
 
    '''Evaluation of the results - Single train test split'''
    accuracyOfSingleSplit = accuracy_score(classLabel_test,pred_classLabel)*100
    return accuracyOfSingleSplit

def predictKFoldCV(classifierBuildT,features,classLabel):
    kfolds = KFold(n_splits=10,random_state=100)
    kfolds.get_n_splits(features,classLabel)
   
    for train, test in kfolds.split(features,classLabel):
        feature_train, feature_test = features[train],features[test]
       
        class_train, class_test = classLabel[train], classLabel[test]
       
    
    classfrDT = classifierBuildT.fit(feature_train,class_train)
   
    '''Save the built model to a file and re-load from file(serialization and de-serialization)'''
    savedModel = 'DTree_model.sav'
    pickle.dump(classfrDT, open(savedModel, 'wb'))
    loadDT_model = pickle.load(open(savedModel, 'rb'))
   
    '''Use the fit model to predict the class on test data'''
    pred_class = loadDT_model.predict(feature_test)
   
    '''Evaluation of the results - 10 fold C.V.'''
    accuracyOfKFoldCV = accuracy_score(class_test,pred_class)*100
    return accuracyOfKFoldCV
    
def predictStratifiedFoldCV(classifierBuildT,features,classLabel):
    stratfolds = StratifiedKFold(n_splits=10,random_state=100)
    stratfolds.get_n_splits(features,classLabel)
   
    for train, test in stratfolds.split(features,classLabel):
        ftre_train, ftre_test = features[train],features[test]
       
        cls_train, cls_test = classLabel[train], classLabel[test]
       
    classfrDT = classifierBuildT.fit(ftre_train,cls_train)
   
    '''Save the built model to a file and re-load from file(serialization and de-serialization)'''
    svdModel = 'DTr_model.sav'
    pickle.dump(classfrDT, open(svdModel, 'wb'))
    loadDT_model = pickle.load(open(svdModel, 'rb'))
   
    '''Use the fit model to predict the class on test data'''
    pred_cls = loadDT_model.predict(ftre_test)
   
    '''Evaluation of the results - stratified 10 fold C.V.'''
    accuracyOfStratifiedFoldCV = accuracy_score(cls_test,pred_cls)*100
    return accuracyOfStratifiedFoldCV

'''Extract data from CSV file onto a data frame'''
dataFrame = pd.read_csv("C:/Masters/DKE/DataSetAssignment/car.csv")

'''Check if ur dataset is balanced/imbalanced'''
print(dataFrame)
targetVar_count = dataFrame.target.value_counts()
# print(targetVar_count)
print('Class label 1', targetVar_count[0])
print('Class label 2', targetVar_count[1])
print('Class label 2', targetVar_count[2])
print('Class label 2', targetVar_count[3])

'''Creating copy of the dataFrame since u r going to modify it'''
dataFrameCopy = dataFrame.copy()
dataFrameCopy = mapStringToInt(dataFrameCopy)
 
'''Data slicing'''
features = dataFrameCopy.values[:,0:6]
classLabel = dataFrameCopy.values[:,6]
 
'''Train-test split'''
features_train, features_test, classLabel_train, classLabel_test = train_test_split( features, classLabel, test_size = 0.3, random_state = 100)
 
classifierBuildT = tree.DecisionTreeClassifier()
accuracyOfSingleSplit = predictCLSingleTrainTest(classifierBuildT,features_train,classLabel_train,features_test,classLabel_test)
print("Accuracy of single train-test split is! ", accuracyOfSingleSplit)
   
'''Using k(10) fold cross validation '''
 
accuracyOfKFoldCV = predictKFoldCV(classifierBuildT,features,classLabel)
print("Accuracy of 10 fold CV is! ", accuracyOfKFoldCV)
 
'''Stratified k fold/Re-sampling on the data'''
 
accuracyOfStratifiedFoldCV = predictStratifiedFoldCV(classifierBuildT,features,classLabel)
print("Accuracy of stratified CV is! ", accuracyOfStratifiedFoldCV)
  
