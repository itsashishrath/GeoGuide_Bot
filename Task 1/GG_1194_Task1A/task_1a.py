'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ 1194 ]
# Author List:		[ Keshav Joshi, Ashish Rathore, Disha Chhabra, Dev Goyal]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim


###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
# Writing device agnostic code and code for reproducabiltiy and using 42 as manual seed
seed = torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################

def data_preprocessing(task_1a_dataframe):


	''' 
	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						  Pandas dataframe read from the provided dataset 	
	
	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to 
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
	

	categorical_columns = task_1a_dataframe.select_dtypes(include=['category']).columns

	# Encoding the data and particular coloumns and using various encodings like one hot and label
	task_1a_dataframe = pd.get_dummies(task_1a_dataframe, columns=categorical_columns)

	label_encoder = LabelEncoder()
	task_1a_dataframe['Education'] = label_encoder.fit_transform(task_1a_dataframe['Education'])
	task_1a_dataframe['City'] = label_encoder.fit_transform(task_1a_dataframe['City'])
	task_1a_dataframe['Gender'] = label_encoder.fit_transform(task_1a_dataframe['Gender'])
	task_1a_dataframe['JoiningYear'] = label_encoder.fit_transform(task_1a_dataframe['JoiningYear'])
	task_1a_dataframe['PaymentTier'] = label_encoder.fit_transform(task_1a_dataframe['PaymentTier'])
	task_1a_dataframe['ExperienceInCurrentDomain'] = label_encoder.fit_transform(task_1a_dataframe['ExperienceInCurrentDomain'])
	task_1a_dataframe['Age'] = label_encoder.fit_transform(task_1a_dataframe['Age'])
	task_1a_dataframe['EverBenched'] = label_encoder.fit_transform(task_1a_dataframe['EverBenched'])

	task_1a_dataframe['LeaveOrNot'] = label_encoder.fit_transform(task_1a_dataframe['LeaveOrNot'])
	##########################################################

	return task_1a_dataframe

def identify_features_and_targets(encoded_dataframe):
	'''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	
	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''
	
	#################	ADD YOUR CODE HERE	##################
	# Selction of data and distributing and specifing it into features and target
	features = encoded_dataframe.drop(['LeaveOrNot', 'EverBenched'], axis=1)
	target = encoded_dataframe['LeaveOrNot']
	features_and_targets=[features,target]
	##########################################################

	return features_and_targets


def load_as_tensors(features_and_targets):

	''' 
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training 
	and validation, and then load them as as tensors. 
	Training of the model requires iterating over the training tensors. 
	Hence the training sensors need to be converted to iterable dataset
	object.
	
	Input Arguments:
	---
	`features_and targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label
	
	Returns:
	---
	`tensors_and_iterable_training_data` : [ list ]
											Items:
											[0]: X_train_tensor: Training features loaded into Pytorch array
											[1]: X_test_tensor: Feature tensors in validation data
											[2]: y_train_tensor: Training labels as Pytorch tensor
											[3]: y_test_tensor: Target labels as tensor in validation data
											[4]: Iterable dataset object and iterating over it in 
												 batches, which are then fed into the model for processing

	Example call:
	---
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	'''

	#################	ADD YOUR CODE HERE	##################
	X = torch.Tensor(features_and_targets[0].values)
	y = torch.Tensor(features_and_targets[1].values)

	# spliting the data inot test and traing usign sklearn train_test_split and keeping testing data at 25% and rest 75% data for training
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# for putting data in tensors and then can you normal data for plotting
	X_train_tensor = X_train
	X_test_tensor = X_test
	y_train_tensor = y_train
	y_test_tensor = y_test

	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	# Creting batch size as data is large and we have limited computational power and small batch size will lead to genralized result and increase accuracy
	batch_size = 64
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	

	return [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_dataloader]
	##########################################################

class Salary_Predictor(nn.Module):
	'''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		'''
		Define the type and number of layers
		'''
		#######	ADD YOUR CODE HERE	#######
		# Here we are defining the neural network struture with is of 4 layer and have 3 hidden layes and we are using relu activation in between hidden layers and sigmoid activation in output layer.
		self.layer_1 = nn.Linear(7, 256) 
		self.layer_2 = nn.Linear(256, 128)
		self.layer_out = nn.Linear(128, 1) 
		self.relu = nn.ReLU()
		self.prelu = nn.PReLU(1)
		self.sigmoid =  nn.Sigmoid()
		self.dropout = nn.Dropout(p=0.5)
		###################################	
	# it is an important function as we are then the working of our neural network
	def forward(self, x):
		x = self.relu(self.layer_1(x))
		x = self.prelu(self.layer_2(x))
		x = self.dropout(x)
		x = self.sigmoid(self.layer_out(x))
		return x



def model_loss_function():
	'''
	Purpose:
	---
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	#################	ADD YOUR CODE HERE	##################
	# Importatnt to take track of loss and learning for then to minimize it.
	return nn.BCELoss()
	##########################################################
	

def model_optimizer(model):
	'''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
	#################	ADD YOUR CODE HERE	##################
	# Important for making accurate prediciton and adam is getter than SGD
	return optim.Adam(model.parameters(), lr=0.001)
	##########################################################

def model_number_of_epochs():
	'''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
	#################	ADD YOUR CODE HERE	##################
	# Best suited for our model
	number_of_epochs=50
	##########################################################

	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	'''
	Purpose:
	---
	All the required parameters for training are passed to this function.

	Input Arguments:
	---
	1. `model`: An object of the 'Salary_Predictor' class
	2. `number_of_epochs`: For training the model
	3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors
	4. `loss_function`: Loss function defined for the model
	5. `optimizer`: Optimizer defined for the model

	Returns:
	---
	trained_model

	Example call:
	---
	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

	'''	
	#################	ADD YOUR CODE HERE	##################
	# genral approch and putting model to training mode of efficient training

	for epoch in range(number_of_epochs):
		
		for batch_x, batch_y in tensors_and_iterable_training_data[4]:
			model.train()
			y_preds = model(batch_x)
			optimizer.zero_grad()
			batch_y = batch_y.view(-1,1)
			loss = loss_function(y_preds, batch_y)
			loss.backward()
			optimizer.step()
	##########################################################

	return model

def validation_function(trained_model, tensors_and_iterable_training_data):
	'''
	Purpose:
	---
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	Input Arguments:
	---
	1. `trained_model`: Returned from the training function
	2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors

	Returns:
	---
	model_accuracy: Accuracy on the validation dataset

	Example call:
	---
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

	'''	
	#################	ADD YOUR CODE HERE	##################
	X_test_tensor=tensors_and_iterable_training_data[1]
	y_test_tensor=tensors_and_iterable_training_data[3]
	train_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoader for batching
	batch_size = 64
	dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	correct = 0
	total = 0
	# putting the model in eval mode and finding the accuracy of our trained model and using inference_mode for better stability
	trained_model.eval()
	with torch.inference_mode():
		for X_test_tensor, y_test_tensor in dataloader:
			print(f"Batch Size: {X_test_tensor.size(0)}")
			y_preds_training = trained_model(X_test_tensor)
			predicted = (y_preds_training > 0.5).float().view(-1)  # Convert to binary predictions
			print("predicted:", predicted)
			total += y_test_tensor.size(0)
			matches = (predicted == y_test_tensor)

			total_matches = matches.sum().item()
			print("Total Matches:", total_matches)

			correct +=total_matches

			print(f"Batch Correct: {correct}")
			print(f"Batch Total: {total}")

	accuracy = 100 * correct / total
	##########################################################

	return accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")