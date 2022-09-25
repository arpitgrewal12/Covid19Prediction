"""
CMPT 459 Milestone 2
Elia Karimi
301369976
Decision Tree and Random Forest Classifier on COVID-19 Dataset
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import pipeline

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
#import category_encoders as ce
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from pickle import dump
from pickle import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE 


def main():
	df = pd.read_csv('Milestone-2_cleaned-data.csv', sep=',', parse_dates=['date_confirmation'])

	del df['Lat']
	del df['Long_']
	del df['source']
	del df['Combined_Key']
	del df['Last_Update']

	#df[df['Confirmed'].isna()]

	df = df.dropna(axis=0)

	#For testing purposes, choose random subset of the data
	def PartitionData(data, numberClusters):   
	    subsetData = data.sample(numberClusters, axis=0, replace=False)
	    return subsetData
	df2 = PartitionData(df,5000)


	df['date_confirmation']=df['date_confirmation'].map(dt.datetime.toordinal)

	df = pd.get_dummies(df,columns=['sex','province','country']) #One-hot encode the data using pandas get_dummies


	# # 2.1 - Splitting the Dataset

	X = df.drop(columns='outcome') #Make the feature matrix (X)
	#X = ce.OneHotEncoder(use_cat_names=True).fit_transform(X) #Encode categorical features
	y = df['outcome'].replace({'hospitalized':0, 'nonhospitalized':1,'recovered':2,'deceased':3}) #Make encoded target vector(y)

	#y = labelledDf['outcome'].values #Extract the outcome as y values

	#Partition data into training and validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)
	#df['outcome'].value_counts(normalize=True)


	#print("Baseline Prediction")
	#Use the mode of the class to create our baseline prediction.
	#majority_class = y_train.mode()[0]
	#baseline_predictions = [majority_class] * len(y_train)

	#majority_class_accuracy = accuracy_score(baseline_predictions,y_train)
	#print(majority_class_accuracy)

	"""
	print("Training RFECV Model")
	m = RFECV(RandomForestClassifier(n_estimators=100,max_depth=10, min_samples_leaf=10),scoring='accuracy')
	m.fit(X,y)
	m.predict(X)

	dump(rfModel, open('model.pkl', 'wb'))


	#Load the model
	model = load(open('model.pkl', 'rb'))

	#Print Random Forest Model accuracy score
	print("Train Random Forest Score=", model.score(X_train, y_train))
	print("Validate Random Forest Score=",model.score(X_test, y_test))
	"""

	#model = make_pipeline(
	 #   SimpleImputer(strategy='mean'), # impute missing values
	  #  MinMaxScaler(),                 # scale each feature to 0-1
	   # PolynomialFeatures(degree=3, include_bias=True),
	    #LinearRegression(fit_intercept=False)
	#)

	#-----------------Create a Decision Tree Classifer and Train It-----------------
	#tree = DecisionTreeClassifier(max_depth=1)
	# Fit the model
	#tree.fit(X_train, y_train)
	# Visualize the tree
	#dot_data = export_graphviz(tree, out_file=None, feature_names=X_train.columns, class_names=['Poisonous', 'Edible'], filled=True, impurity=False, proportion=True)
	#graphviz.Source(dot_data)

	#Print Decision Tree Model accuracy score
	#print("Train Random Forest Score=", tree.score(X_train, y_train))
	#print("Validate Random Forest Score=",tree.score(X_test, y_test))
	

	#-----------------Recursive Feature Elimination-----------------


	#RFE is popular because it is easy to configure and use and because it is effective at selecting those features (columns) in a training dataset 
	#that are more or most relevant in predicting the target variable.

	#There are two important configuration options when using RFE: the choice in the number of features to select and the choice of the algorithm used to help choose features.
	#Feature selection refers to techniques that select a subset of the most relevant features (columns) for a dataset. 
	#Fewer features can allow machine learning algorithms to run more efficiently (less space or time complexity) and be more effective. 
	#Some machine learning algorithms can be misled by irrelevant input features, resulting in worse predictive performance.

	#This is achieved by fitting the given machine learning algorithm used in the core of the model, ranking features by importance, discarding the least important features, and re-fitting the model. 
	#This process is repeated until a specified number of features remains.

	# define the method
	# fit the model
	#rfe.fit(X_train_scaled, y)
	# transform the data
	#X, y = rfe.transform(X, y)

	print("Training RFE + Random Forest Model")

	rfm = RandomForestClassifier(n_estimators=85, max_depth=7, min_samples_leaf=5)

	std_scalar = preprocessing.StandardScaler()

	selector = feature_selection.RFE(rfm)

	pipe_params = [('scaler',std_scalar), ('select', selector),('rf model',rfm)]
	
	pipe = pipeline.Pipeline(pipe_params)
	pipe.fit(X_train,y_train)
	print(pipe.score(X_test,y_test))
	#rfModel = make_pipeline(
	#    StandardScaler(),
	#	RFE(estimator=RandomForestClassifier, n_features_to_select=10),
	#)

	# save the model
	dump(pipe, open('model.pkl', 'wb'))

	#Load the model
	model = load(open('model.pkl', 'rb'))
	y_predicted = model.predict(X_test)

	#Print Random Forest Model accuracy score
	print("Train Random Forest Score=", model.score(X_train, y_train))
	print("Validate Random Forest Score=",model.score(X_test, y_test))

	#Precision: how many selected were correct
	#Recall: how many correct were found?
	print(classification_report(y_valid, y_predicted))

	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	# report performance
	print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
	#-----------------Create a Random Forest Classifer and Train It-----------------
	"""
	print("Training Random Forest Model")
	rfModel = make_pipeline(
	    StandardScaler(),
	    RandomForestClassifier(n_estimators=85, max_depth=7, min_samples_leaf=5)
	)

	rfModel.fit(X_train, y_train)
	# save the model
	dump(rfModel, open('model.pkl', 'wb'))
	# save the scaler
	# define scaler
	#scaler = MinMaxScaler()
	# fit scaler on the training dataset
	#scaler.fit(X_train)
	# transform the training dataset
	#X_train_scaled = scaler.transform(X_train)
	#dump(scaler, open('scaler.pkl', 'wb'))

	#Load the model
	model = load(open('model.pkl', 'rb'))
	y_predicted = model.predict(X_test)

	#Print Random Forest Model accuracy score
	print("Train Random Forest Score=", model.score(X_train, y_train))
	print("Validate Random Forest Score=",model.score(X_test, y_test))

	#Precision: how many selected were correct
	#Recall: how many correct were found?
	print(classification_report(y_valid, y_predicted))


	# load the scaler
	#scaler = load(open('scaler.pkl', 'rb'))
	"""
if __name__ == '__main__':
    main()
