from random import random
from math import exp
import pandas as pd
import numpy as np

LEARN_RATE = 0.01

def generateNeuralNet(numLayers, lenLayer):

	ann = []

	for i in range(numLayers):
		layer = []
		for j in range(lenLayer):
			layer.append( [ 0, [ random() for k in range(lenLayer) ], 1 ] )
		ann.append(layer)

	ann.append( [ [ 0, [ random() for k in range(lenLayer) ], 1 ] ] )

	return ann

def sigm(x):
	return ( 1 / ( 1 + exp(-x) ) )

def forward(ann, values, numLayers, lenLayer):

	for i in range(lenLayer):
		ann[0][i][0] = values[i]
	for i in range(1, numLayers):
		for j in range(lenLayer):

			summation = 0

			for k in range(lenLayer):
				summation += ann[i][j][1][k] * ann[i-1][k][0]

			ann[i][j][0] = sigm(summation)


	summation = 0

	for k in range(lenLayer):
		#print(ann[numLayers][0][1][k])
		summation += ann[numLayers][0][1][k] * ann[numLayers-1][k][0]

	ann[numLayers][0][0] = sigm(summation)
	#print("Return - %.5f"%ann[numLayers][0][0])

	return ann

def backward(ann, expecValue, numLayers, lenLayer):

	#print("Return - %.5f"%ann[numLayers][0][0])
	#print("Class - %.5f"%expecValue)

	difError = 2*( ann[numLayers][0][0] - expecValue ) * ( 1 - ann[numLayers][0][0] ) * ann[numLayers][0][0]
	ann[numLayers][0][2] = difError

	for i in range(lenLayer):
		ann[numLayers][0][1][i] -= LEARN_RATE * difError * ann[numLayers-1][i][0]

	for j in range(1, numLayers, -1):

		for i in range(lenLayer):

			sum_dif = 0
			for u in ann[j+1]:
				sum_diff += u[2] * ( u[1][i] + LEARN_RATE * u[2] * ann[j][i][0] )

			difError = sum_diff * ( ( 1 -  ann[j][i][0] ) * ann[j][i][0] )
			ann[j][i][2] = difError

			for k in range(lenLayer):
				ann[j][i][1][k] -= LEARN_RATE * difError * ann[j-1][k][0] 

	return ann


def train(ann, numLayers, lenLayer, trainData, ncol):

	for i in range(trainData.shape[0]):
		ann = forward(ann, trainData.iloc[i, 0:ncol-1], numLayers, lenLayer)
		ann = backward(ann, trainData.iloc[i, ncol-1], numLayers, lenLayer)

	return ann

def test(ann, numLayers, lenLayer, testData, ncol):

	medium_abs_error = 0

	for i in range(testData.shape[0]):
		ann = forward(ann, testData.iloc[i, 0:ncol-1], numLayers, lenLayer)
		print("Class - %.5f"%testData.iloc[i, ncol-1])
		print("Return - %.5f"%ann[numLayers][0][0])
		medium_abs_error += abs( testData.iloc[i, ncol-1] - ann[numLayers][0][0] )

	medium_abs_error /= testData.shape[1]
	return medium_abs_error


#Data

data = pd.read_csv("EyeState/eye.csv")

numLayers = 401
lenLayer = data.shape[1] - 1

ann = generateNeuralNet(numLayers, lenLayer)

#Train
ann = train(ann, numLayers, lenLayer, data.iloc[ 0 : 2 * data.shape[0] // 3], data.shape[1])

print(test(ann, numLayers, lenLayer, data.iloc[2 * data.shape[0] // 3 : ], data.shape[1]))
