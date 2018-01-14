from __future__ import division
from nn import NeuralNetwork
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_data():
	data_path = 'Bike-Sharing-Dataset/hour.csv'
	rides = pd.read_csv(data_path)
	return rides

def categorize_data(df):
	''' Clean, categorize, and scale data

		Args
		----
		df: DataFrame containing features/targets
	'''
	rides = df

	# One hot encoding
	dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
	for each in dummy_fields:
	    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
	    rides = pd.concat([rides, dummies], axis=1)

	fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
	                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
	data = rides.drop(fields_to_drop, axis=1)
	
	target_fields = ['cnt', 'casual', 'registered']
	quant_features = ['casual','registered', 'cnt', 'temp', 'hum', 'windspeed']
	
	# Store scalings in a dictionary so we can convert back later
	scaled_features = {}
	for each in quant_features:
	    mean, std = data[each].mean(), data[each].std()
	    scaled_features[each] = [mean, std]
	    data.loc[:, each] = (data[each] - mean)/std
	features, targets = data.drop(target_fields, axis=1), data[target_fields]

	return features, targets, scaled_features

def create_NN(X, y):	
	''' Initialize neural network and train
		
		Args
		----
		X: 2D features array
		y: 1D target array
	'''
	X_train, X_test, y_train, y_test = train_test_split(X,y)
	feature_count = X_train.shape[1] 
	
	nn = NeuralNetwork(feature_count,13,1,1.0,2500)		## HYPERPARAMETERS
	# nn = NeuralNetwork(feature_count)		
	nn.iterate(X_train, X_test, y_train, y_test)

	return nn

def plotter(nn, scaled_features,features,targets,df):
	''' Plot losses and predictions

		Args
		----
		nn 				: Trained neural network
		scaled_features : Dict of mean/std values for scaled data 
		features 		: 2D features array
		targers			: 1D target array
		df 				: DataFrame for date labels
	'''
	plt.plot(nn.losses['train'], label='Training loss')
	plt.plot(nn.losses['test'], label='Test loss')
	plt.legend()
	_ = plt.ylim()
	plt.savefig('mse.png')

	fig, ax = plt.subplots(figsize=(8,4))
	mean, std = scaled_features['cnt']
	predictions = nn.run(features).T*std + mean
	#change ax to plot
	ax.plot(predictions[0], 'bs', label='Prediction', linewidth = 1)#,'bo')
	ax.plot((targets['cnt']*std + mean).values, 'r+', label='Data', linewidth = 1)#,'r+	')
	ax.set_xlim(right=len(predictions))
	ax.legend()
	dates = pd.to_datetime(df.loc[targets.index]['dteday'])
	dates = dates.apply(lambda d: d.strftime('%b %d'))
	ax.set_xticks(np.arange(len(dates))[12::24])
	_ = ax.set_xticklabels(dates[12::24], rotation=45)
	plt.savefig('ride_prediction.png')

if __name__ == '__main__':
	df = read_data()
	features, targets, scaled_features = categorize_data(df)
	nn = create_NN(features, targets)
	plotter(nn, scaled_features, features, targets, df)