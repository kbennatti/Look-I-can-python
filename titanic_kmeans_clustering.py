# assigned task: use titanic data set to implement k means clustering. decide on a value for k based on visual assessment
# of a dendrogram
# note: the commented steps are a simplified version of the instructions given by my professor
# all code is my own except for the dendrogram plot portion which was partially copied from class notes

import json
import random
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import numpy as np

f = open('/home/datascience/Desktop/ass3/titanic.json', 'r')
d = json.load(f)

# create array of zeros, 12 across, 891 lines
dArray = [[0 for x in range(0, 6)] for y in range(0, len(d))]
# replace zeros with features from json file
for i in range(0, len(d)):
    if d[i]['Age'] == '':
        dArray[i][0] = None
    else:
        dArray[i][0] = float(d[i]['Age'])
    dArray[i][1] = float(d[i]['Fare'])
    # make siblings/spouse and parent/child into companions count
    dArray[i][2] = float(d[i]['SiblingsAndSpouses']) + float(d[i]['ParentsAndChildren'])
    # encode embarked location and sex -> i think these should be dummy coded instead
    if d[i]['Embarked'] == 'C':
        dArray[i][3] = 1.
    elif d[i]['Embarked'] == 'Q':
        dArray[i][3] = 2.
    elif d[i]['Embarked'] == 'S':
        dArray[i][3] = 3.
    elif not d[i]['Embarked']:
        dArray[i][3] = None
    else:
        print 'uh oh', i, d[i]['Embarked']
    if d[i]['Sex'] == 'male':
        dArray[i][4] = 0
    elif d[i]['Sex'] == 'female':
        dArray[i][4] = 1
    elif not d[i]['Sex']:
        dArray[i][4] = None
    else:
        print 'uh oh', i, d[i]['Sex']
    # add the target
    dArray[i][5] = float(d[i]['Survived'])

# check missing values
missing = []
for row in dArray:
    if None in row:
        missing.append(row)
print 'Number observations with missing values is: ', len(missing)
print 'That\'s ', (len(missing)/float(len(dArray)))*100, '% of the total'
print 'If we remove observations with missing values we have ', len(dArray) - float(len(missing)), \
    ' observations remaining'

# will remove observations with missing values because imputing values of age may add bias and 712 observations
# is sufficient

# remove observations with missing values
dataNoMiss = []
for row in dArray:
    if None not in row:
        dataNoMiss.append(row)

print len(dataNoMiss)

# normalize features
dataNoMiss = np.array(dataNoMiss)

# make a list of names
featureNames = ['age','fare','companions_count', 'embarked_location', 'sex', 'survived']
# figure out max and mins so normailization is possible
dicMin = {}
dicMax = {}
t = 0
while t < len(dataNoMiss[0]):
    minVal = dataNoMiss[0][t]
    maxVal = dataNoMiss[0][t]
    for row in range(0, len(dataNoMiss)):
        if dataNoMiss[row][t] < minVal:
            minVal = dataNoMiss[row][t]
        if dataNoMiss[row][t] > maxVal:
            maxVal = dataNoMiss[row][t]
        dicMin[featureNames[t]] = minVal
        dicMax[featureNames[t]] = maxVal
    t += 1
# normalize

for i in range(0, len(dataNoMiss)):
    for j in range(0, 4): # don't need to normalize sex
        dataNoMiss[i][j] = (dataNoMiss[i][j] - dicMin[featureNames[j]]) / (dicMax[featureNames[j]] - dicMin[featureNames[j]])

# create a dendrogram and determine a threshold

Z = linkage(dataNoMiss[:, 0:5], method='ward', metric='euclidean') # distance between clusters and metric
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Instance Index')
plt.ylabel('Distance')
# creates a dendrogram hierarchial plot
dendrogram(Z, leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=3.,  # font size for the x axis labels
            )
# visually, 9 seems like a good threshold because after that the distance between cluster breaks becomes quite large
plt.axhline(y=9, color='black')
plt.show()

# at a threshold of 9 (euclidean distance), there are 3 clusters


# implement k means clustering where k = 3

# create function to calculate cluster mean
def clusterMean(cluster):
    sumAge = 0
    sumFare = 0
    sumCompanions = 0
    sumEmbarked = 0
    sumSex = 0
    for instance in cluster:
        sumAge += instance[0]
        sumFare += instance[1]
        sumCompanions += instance[2]
        sumEmbarked += instance[3]
        sumSex += instance[4]
    sums = [sumAge, sumFare, sumCompanions, sumEmbarked, sumSex]
    mean = [float(Sum)/len(cluster) for Sum in sums]
    return mean


# assign centroids randomly in feature space
q1 = [random.uniform(0, 1) for number in xrange(5)]
q2 = [random.uniform(0, 1) for number in xrange(5)]
q3 = [random.uniform(0, 1) for number in xrange(5)]

centroids = [q1, q2, q3]

t = 0
j = 0
oldCluster1 = 1
oldCluster2 = 1
oldCluster3 = 1
while j<10 and t<1: # repeat until cluster assignment stops changing
    cluster1 = []
    cluster2 = []
    cluster3 = []
    # assign each instance to cluster with closest centroid
    for instance in dataNoMiss:
        d1 = distance.euclidean(instance[0:5], q1)
        d2 = distance.euclidean(instance[0:5], q2)
        d3 = distance.euclidean(instance[0:5], q3)
        if min(d1, d2, d3) == d1:
            cluster1.append(instance)
        elif min(d1, d2, d3) == d2:
            cluster2.append(instance)
        elif min(d1, d2, d3) == d3:
            cluster3.append(instance)
        else:
            print 'something went wrong'

    print len(cluster1), len(cluster2), len(cluster3), len(cluster1 + cluster2 + cluster3)

    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    cluster3 = np.array(cluster3)

    # check if cluster assignment stopped changing
    if np.array_equal(cluster1, oldCluster1) and np.array_equal(cluster2, oldCluster2): # do i need to change this back for lists?
        t = 1

    # recompute cluster centroid
    q1 = clusterMean(cluster1)
    q2 = clusterMean(cluster2)
    q3 = clusterMean(cluster3)

    print q1, q2, q3

    oldCluster1 = cluster1[:]
    oldCluster2 = cluster2[:]
    oldCluster3 = cluster3[:]

    if j is 0 or j is 5 or j is 10 or t is 1:
        for i in range(0, 5):
            for k in range(0, 5):
                if i < k:
                    plt.plot(cluster1[:, i], cluster1[:, k], 'gx')
                    plt.plot(cluster2[:, i], cluster2[:, k], 'bx')
                    plt.plot(cluster3[:, i], cluster3[:, k], 'kx')
                    plt.plot(q1[i], q1[k], 'go', label='centroid 1')
                    plt.plot(q2[i], q2[k], 'go', label='centroid 2', color='blue')
                    plt.plot(q3[i], q3[k], 'go', label='centroid 3', color='black')
                    plt.title(featureNames[i] + ' vs. ' + featureNames[k])
                    plt.xlabel(featureNames[i])
                    plt.ylabel(featureNames[k])
                    plt.legend()
                    plt.savefig('%s_vs_%s_%i.png' % (featureNames[i], featureNames[k], j))
                    plt.clf()
    j += 1


#check the proportion of survived in each cluster
countTarget1 = 0
countTarget2 = 0
countTarget3 = 0

for instance in cluster1:
    if instance[5] == 1:
        countTarget1 += 1
for instance in cluster2:
    if instance[5] == 1:
        countTarget2 += 1
for instance in cluster3:
    if instance[5] == 1:
        countTarget3 += 1

print str((countTarget1/float(len(cluster1))*100)) + '% survived in cluster 1'
print str((countTarget2/float(len(cluster2))*100)) + '% survived in cluster 2'
print str((countTarget3/float(len(cluster3))*100)) + '% survived in cluster 3'
