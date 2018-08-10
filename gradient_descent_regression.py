# assigned task: create my own gradient descent linear regression model instead of using a python package
# we were instructed to avoid using built-in functions and packages other than those imported below
# the commented steps are a simplified version of the instructions given by my professor
# all code is my own
from sklearn.datasets import load_boston
import math as math
import numpy as np
import matplotlib.pyplot as plt
boston = load_boston()
data = boston.data
target = boston.target
rowsInData = data.shape[0]
colInData = data.shape[1]

#step 1: normalize features

#1a: Figure out max and min values and add them to a dictionary without using built-in functions
dicMin = {}
dicMax = {}
t=0
while t<13: 
	minVal = data[0][t]
	maxVal = data[0][t]

	for row in range(0,rowsInData):
		if data[row][t] < minVal:
			minVal = data[row][t]
		if data[row][t] > maxVal:
			maxVal = data[row][t]
		dicMin[t] = minVal
		dicMax[t] = maxVal
	t+=1

#step1b: normalize features using min and max

dataNorm = data.copy()
t=0
while t<13:
	for row in  range(0,rowsInData):
		dataNorm[row][t] = (dataNorm[row][t] - dicMin[t])/(dicMax[t]-dicMin[t])
	t+=1

#step1c: find min and max of target and normalize
minValTar = min(target)
maxValTar = max(target)

targetNorm = target.copy()

for row in  range(0,target.shape[0]):
	targetNorm[row] = (targetNorm[row] - minValTar)/(maxValTar-minValTar)
	
print 'Check that targetNorm values are between 0 and 1, minimum value is ', min(targetNorm), 'maximum is ', max(targetNorm)

#step2: split up boston data into training and validation (90:10)
#start by combining the features and the target
fullTable=[]   

for i in range(0,rowsInData):
	feature = dataNorm[i][0:13]
	targets = targetNorm[i]
	row = [feature, targets]
	fullTable.append(row)

#then separate by assigning every 10th row to the test set
test = []
train = []
for i in range(0, len(fullTable)):
	if i%10 ==0:
		test.append(fullTable[i])
	else:
		train.append(fullTable[i])

#keep separate normalized target vector
testTarget = []
trainTarget = []
for i in range(0, len(targetNorm)):
	if i%10 ==0:
		testTarget.append(targetNorm[i])
	else:
		trainTarget.append(targetNorm[i])


#print out the percentage of rows assigned to test vs. train
print 'Number values assigned to test is', len(test),' and % assigned to test is ', (len(test)/float(len(fullTable)))*100, '%'


print 'Number values assigned to train is', len(train), ' and % assigned to train is ',(len(train)/float(len(fullTable)))*100, '%'


#step3: create an RMSE function

def rmse(n, yi, yhat):
	errorlist = []
	for i in range(0,len(yhat)):
		errorsq = (yi[i]-yhat[i])**2
		errorlist.append(errorsq)
	return math.sqrt(sum(errorlist)/float(n))

#step4: create list of learning rates
learning = [1]
for i in range(0,4):
	new = learning[i]/float(10)
	learning.append(new)

print 'List of learning rates: ', learning

#step5: create a regression function

def yHat(b0, biList, xiList):
	terms = []
	for i in range(0, len(xiList)):
		term = biList[i]*xiList[i]
		terms.append(term)
	return b0 + sum(terms)


#step6: do some gradient descent

#create list of list to store 10 errors from each epoch for each learning rate
rmsErrorsList = [[0 for x in range(10)] for y in range(len(learning))]
#run through each learning rate 10 times, using gradient descent and calculate RMSE
for learningIndex in range(0,len(learning)):
	b0=0.0
	b = [0 for x in range(13)]
	epoch = 0
	while epoch < 10:
		predictions = []
		for i in range(0, len(train)):
			error = yHat(b0, b, train[i][0]) - train[i][1]
	  		b0 = b0 - learning[learningIndex]*error
		for j in range(0, len(b)):
			b[j] = b[j]- learning[learningIndex]*error*train[i][0][j]
		for k in range(0, len(train)):
			prediction = yHat(b0,b,train[k][0])
			predictions.append(prediction)	
		RMSE = rmse(len(train), trainTarget, predictions)
		rmsErrorsList[learningIndex][epoch] = RMSE
		epoch+=1

print 'Here\'s the list of RMSEs for each learning rate: ', rmsErrorsList

#plot that sucker
epochs = range(1,11)

for i in range(0, len(learning)):
	plt.plot(epochs, rmsErrorsList[i])
	plt.title('RMSE Plot for Learning Rate '+str(learning[i]))
	plt.xlabel('Epoch')
	plt.ylabel('RMSE')
	plt.show()


#Conclusions and comments: From the plots I can see that by using a learning rate of 1, RMSE gets worse, not better over time. However, below 1, the higher learning rates learn faster i.e. get to lower RMSEs faster than lower value learning rates.
