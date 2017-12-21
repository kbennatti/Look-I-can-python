# assigned task: use a data set that contains information about whether customers are early or late adopters to build a decision tree and evaluate
# note: the commented steps are a simplified version of the instructions given by my professor
# all code is my own except for the section labeled "encode" which was partially copied from stack overflow

import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random as rand
# notes on sklearn decision trees: does not handle missing values, fine with categorical features but must convert into dummy variables first

# import data
dataOriginal = open('/home/datascience/Desktop/ass2/a1_dataset.dat', 'r').readlines()

data = []

for line in dataOriginal:
	row = line.strip().split(';')	
	data.append(row)

# clean data

# remove various titles from data
dataClean = []
titles = []
for i in range(0,len(data)):
	if 'Gender' not in data[i] and len(data[i]) !=0:
		dataClean.append(data[i])
	else:
		titles.append(data[i])
print 'This many titles were removed: ', len(titles)

#get rid of missing values
dataNoMiss = []
dataMiss = []

for row in dataClean:
	rowContainsMissing = []
	for element in row:
		if not element:
			rowContainsMissing.append(dataClean.index(row))
	if rowContainsMissing:
		dataMiss.append(row)
	else:
		dataNoMiss.append(row)

print 'This many rows with missing data were removed: ', len(dataMiss)
	

# set Late and Very Late adopters equal to 1 and everyone else equal to 0 at sublist index 9
for i in range(0, len(dataNoMiss)):
	if "Late" in dataNoMiss[i][9]:
		dataNoMiss[i][9] = 1      
		#print 'Late ', dataClean[i]
	else:
		dataNoMiss[i][9] = 0

#separate features from target
targets = []
features = []

for i in range(0, len(dataNoMiss)):
	targets.append(dataNoMiss[i][9])
	features.append(dataNoMiss[i][:9])

#get rid of the ID because that's not a helpful model input
featuresNoID = []
for i in range(0, len(features)):
		featuresNoID.append(features[i][1:9])

features = featuresNoID

#set up lists containing each class in order to feed into the encoder
featuresList = [[0 for x in range(0,len(features))] for y in range(0,len(features[0]))]
for j in range(0,len(features[0])):
	for i in range(0,len(features)):
		featuresList[j][i]=features[i][j]
#print it to make sure I did it right
for i in range(0,len(featuresList)):
	print featuresList[i][0]
#set up a list to put the integer values into
integersList = [[0 for x in range(0,len(featuresList[0]))] for y in range(0,len(featuresList))]
#encode
label_encoder = LabelEncoder()
for i in range(0,1) + range(2,len(featuresList)): #avoid hot coding age at index 1
	integer_encoded = label_encoder.fit_transform(featuresList[i])
	integersList[i] = integer_encoded
#replace the strings in features with integers
for j in range(0,len(integersList[0])):
	for i in range(0,1) + range(2,len(integersList)):
		features[j][i] = integersList[i][j]

print features[0:3]

#check to make sure the elements of the dataset are the right length
for i in range(0, len(features)):
	if len(features[i]) !=8:
		print i, len(features[i]) #if no print out, we're good!

#separate target into training (90%) and testing (10%)
targetsTest = []
targetsTrain = []
for i in range(0, len(targets)):
	if i%10 ==0:
		targetsTest.append(targets[i])
	else:
		targetsTrain.append(targets[i])
print 'Percentage assigned to targets test is ', (len(targetsTest)/float(len(targets)))*100, '% and length is ', len(targetsTest)
targetsTest = np.array(targetsTest)
targetsTrain = np.array(targetsTrain)

#separate features into training (90%) and testing
featuresTest = []
featuresTrain = []
for i in range(0, len(features)):
	if i%10 ==0:
		featuresTest.append(features[i])
	else:
		featuresTrain.append(features[i])
print 'Percentage assigned to features test is ', (len(featuresTest)/float(len(features)))*100, '% and length is ', len(targetsTest)

featuresTrain = np.array(featuresTrain)
featuresTest = np.array(featuresTest)


#create a decision tree
# for each value in the max leaf nodes list, try each value of the max depth list
# use the tree to predict using test features
#calculate the accuracy of the predictions against actual target values
max_leaf_nodes = [2, 4, 8, 16, 32, 64, 128, 256]
max_depth = [2, 4, 8, 16]
dicAccuracies = {}
for i in range(0, len(max_leaf_nodes)):
	accuracies = []
	for j in range(0, len(max_depth)):
		clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes[i], max_depth = max_depth[j])
		fit = clf.fit(featuresTrain, targetsTrain) # train the model
		predictions = clf.predict(featuresTest)#make predictions
		correct = 0
		incorrect = 0
		for k in range(0, predictions.shape[0]):#calculate accuracy
			if (predictions[k] == targetsTest[k]):
				correct += 1
			else:
				incorrect += 1
		accuracy = float(correct)/(correct+incorrect)
		accuracies.append(accuracy)
	dicAccuracies[max_leaf_nodes[i]] = accuracies

print dicAccuracies

#plot accuracy for each leaf node
for i in range(0, len(max_leaf_nodes)):
	plt.plot(max_depth,dicAccuracies[max_leaf_nodes[i]])
	plt.title('Accuracy at Max Leaf Nodes of '+ str(max_leaf_nodes[i]))
	plt.xlabel('Max Depth')
	plt.ylabel('Accuracy')
	plt.show()


# do the above again  with 50:50 ratio (balanced data)!!
# balanced decision tree code starts here:

#count the current number of late/very late adopters vs. other in the population
countLate = 0
countOther = 0
late = []
others = []
for i in range(0, len(dataNoMiss)):
	if dataNoMiss[i][9]==1:
		countLate+=1
		late.append(dataNoMiss[i])
	elif dataNoMiss[i][9]==0:
		countOther+=1
		others.append(dataNoMiss[i])
	else:
		print 'We have a problem at line ', i

print 'Current percentage of late/very late adopters in the population: ', float(countLate)/len(dataNoMiss)

#create a data set with balanced late vs. other adopters
rand.seed(2222)
sampleIDs = rand.sample(range(0,countOther),countLate)
sampleOthers = []
for i in range(0, len(others)):
	if i in sampleIDs:
		sampleOthers.append(others[i])

balanced = late + sampleOthers #this  dataset has 50% late adopters

#run the balanced set through the decision classifier

targets = []
features = []

for i in range(0, len(balanced)):
	targets.append(balanced[i][9])
	features.append(balanced[i][:9])

#get rid of the ID because that's not a helpful model input
featuresNoID = []
for i in range(0, len(features)):
		featuresNoID.append(features[i][1:9])

features = featuresNoID

#set up lists containing each class in order to feed into the encoder
featuresList = [[0 for x in range(0,len(features))] for y in range(0,len(features[0]))]
for j in range(0,len(features[0])):
	for i in range(0,len(features)):
		featuresList[j][i]=features[i][j]
#print it to make sure I did it right
for i in range(0,len(featuresList)):
	print featuresList[i][0]
#set up a list to put the integer values into
integersList = [[0 for x in range(0,len(featuresList[0]))] for y in range(0,len(featuresList))]
#encode
label_encoder = LabelEncoder()
for i in range(0,1) + range(2,len(featuresList)): #avoid hot coding age at index 1
	integer_encoded = label_encoder.fit_transform(featuresList[i])
	integersList[i] = integer_encoded
#replace the strings in features with integers
for j in range(0,len(integersList[0])):
	for i in range(0,1) + range(2,len(integersList)):
		features[j][i] = integersList[i][j]

#check to make sure the dataset is the right length
for i in range(0, len(features)):
	if len(features[i]) !=8:
		print i, len(features[i]) #if no print out, we're good!

#separate target into training (90%) and testing
targetsTest = []
targetsTrain = []
for i in range(0, len(targets)):
	if i%10 ==0:
		targetsTest.append(targets[i])
	else:
		targetsTrain.append(targets[i])
print 'Percentage assigned to targets test is ', (len(targetsTest)/float(len(targets)))*100, '% and length is ', len(targetsTest)
targetsTest = np.array(targetsTest)
targetsTrain = np.array(targetsTrain)

#separate features into training (90%) and testing
featuresTest = []
featuresTrain = []
for i in range(0, len(features)):
	if i%10 ==0:
		featuresTest.append(features[i])
	else:
		featuresTrain.append(features[i])
print 'Percentage assigned to features test is ', (len(featuresTest)/float(len(features)))*100, '% and length is ', len(targetsTest)

featuresTrain = np.array(featuresTrain)
featuresTest = np.array(featuresTest)

#create a decision tree
# for each value in the max leaf nodes list, try each value of the max depth list
# use the tree to predict using test features
#calculate the accuracy of the predictions against actual target values
max_leaf_nodes = [2, 4, 8, 16, 32, 64, 128, 256]
max_depth = [2, 4, 8, 16]
dicAccuracies = {}
for i in range(0, len(max_leaf_nodes)):
	accuracies = []
	for j in range(0, len(max_depth)):
		clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes[i], max_depth = max_depth[j])
		fit = clf.fit(featuresTrain, targetsTrain) # train the model
		predictions = clf.predict(featuresTest)#make predictions
		correct = 0
		incorrect = 0
		for k in range(0, predictions.shape[0]):#calculate accuracy
			if (predictions[k] == targetsTest[k]):
				correct += 1
			else:
				incorrect += 1
		accuracy = float(correct)/(correct+incorrect)
		accuracies.append(accuracy)
	dicAccuracies[max_leaf_nodes[i]] = accuracies

#plot accuracy for each leaf node
for i in range(0, len(max_leaf_nodes)):
	plt.plot(max_depth,dicAccuracies[max_leaf_nodes[i]])
	plt.title('Accuracy at Max Leaf Nodes of '+ str(max_leaf_nodes[i]))
	plt.xlabel('Max Depth')
	plt.ylabel('Accuracy')
	plt.show()



