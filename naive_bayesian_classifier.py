import numpy as np
import matplotlib.pyplot as plt

# problem: train naive bayes classifier to determine whether a plane is fit to fly

# read in dataset
dataOriginal = open('/home/datascience/Desktop/ass1_DS2/Flying_Fitness.csv', 'r').readlines()

data = []

for line in dataOriginal:
	row = line.strip().split(',')
	data.append(row)

# remove title row
varNames = data[0]

data = data[1:]
# convert strings to floats
dataFloat = [[0 for x in range(0, len(data[0]))] for y in range(0, len(data))]
for i in range(0,len(data)):
    for j in range(0, len(data[0])):
        dataFloat[i][j] = float(data[i][j])

data = dataFloat


# count number of 1s in target (proportion fit to fly)
dicCounts = {}
for j in range(0,len(data[i])):
        # create keys for each variable
    if j not in dicCounts.keys():
        dicCounts[j]=[0,0,0,0]

countedNumbers = [0,1,2,3]
for i in range(0, len(data)):
    for j in range(0,len(data[i])):
        for k in range(0, len(countedNumbers)):
            if data[i][j]==countedNumbers[k]:
                dicCounts[j][countedNumbers[k]] += 1

# calculate the probability of each predictor variable occurring
dicPredictorProb = {}
#create dictionary of zeros

for j in range(0,len(data[i])):
        # create keys for each variable
    if j not in dicPredictorProb.keys():
        dicPredictorProb[j]=[0, 0, 0, 0]

for i in range(0, len(dicCounts)):
    for j in range(0, len(dicCounts[i])):
        dicPredictorProb[i][j]= dicCounts[i][j] / float(len(data))

# count of predictor variables coocurring with 1s in the target variable
dicCondCount = {}
for i in range(0, len(data)):
    for j in range(0,len(data[i])):
        if j not in dicCondCount.keys():
            dicCondCount[j]=[0, 0, 0, 0]

for i in range(0, len(data)):
    for j in range(0,len(data[i])):
        for k in range(0, len(countedNumbers)):
            if data[i][j]== countedNumbers[k] and data[i][1]==1:
                dicCondCount[j][countedNumbers[k]]+=1

# dictionary of conditional probabiilties
dicCond = {}
#create dictionary of zeros
for i in range(0, len(data)):
    for j in range(0,len(data[i])):
        if j not in dicCond.keys():
            dicCond[j]=[0, 0, 0, 0]

for i in range(0, len(dicCondCount)):
    for j in range(0, len(dicCondCount[i])):
        dicCond[i][j]= dicCondCount[i][j] / float(len(data))

# score each observation using the naive bayes classifier
naiveScoreList = [0 for y in range(0, len(data))]
# compute conditional probability of each observation being fit to fly
for i in range(0, len(data)):
    for j in range(2, len(data[i])):
        if naiveScoreList[i] == 0:
            naiveScoreList[i] = dicCond[j][int(data[i][j])]/dicPredictorProb[j][int(data[i][j])]
            #print 'i is', i, 'j is', j, 'and current score is', naiveScoreList[i]
        else:
            naiveScoreList[i] = naiveScoreList[i] * dicCond[j][int(data[i][j])]/dicPredictorProb[j][int(data[i][j])]
            #print 'j is', j, 'and current score is', naiveScoreList[i]
            #print 'predictor', naiveScoreList[i]
    naiveScoreList[i] = naiveScoreList[i]*dicCond[1][1]
    #print 'score', naiveScoreList[i]

# create a list with observation number, target value and predicted score

scoreAndTarget = []
for i in range(0,len(data)):
    scoreAndTarget.append([data[i][0], data[i][1], naiveScoreList[i]])

sortedScore = sorted(scoreAndTarget, key = lambda x: x[2])

#threshold, tpr, fpr
tprAndFpr = []

for i in range(0,len(sortedScore)):
    tp = 0
    fp = 0
    for j in range(0, len(sortedScore)):
        if sortedScore[j][2] > sortedScore[i][2]:
            prediction = 1
            #print i,'threshold is ', sortedScore[i][2], 'score is ', sortedScore[j][2], 'pred is', prediction
        else:
            prediction = 0
            #print i, 'threshold is ', sortedScore[i][2], 'score is ', sortedScore[j][2], 'pred is', prediction
        if prediction and sortedScore[j][1] == 1:
            tp +=1
            #print tp
        if prediction ==1 and sortedScore[j][1] == 0:
            fp +=1
            #print fp
    tprAndFpr.append([sortedScore[i][2],tp/20.0, fp/20.0])

# split up tpr and fpr to plot them

tpr=[]
fpr = []
tprFpr = []

for row in tprAndFpr:
    tpr.append(row[1])
    fpr.append(row[2])
    tprFpr.append(row[1:])

# plot roc curve
#ne3ed to add a x=y line!
x=[0,.5,1]
plt.plot(tpr,fpr, color = 'r')
plt.plot(x,x, color = 'b')
#plt.plot(x, y, color ='b')
plt.title('ROC Curve for Naive Bayesian Clasifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('ROC.png')

