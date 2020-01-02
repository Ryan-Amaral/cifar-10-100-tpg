from tpg.trainer import Trainer
from tpg.agent import Agent
import pickle
import numpy as np
from optparse import OptionParser
import os
import pandas as pd

################################################################################
# command line argument options
################################################################################

parser = OptionParser()

# number of tpg generations to run for
parser.add_option('-g', '--gens', type='int', dest='nGens', default=1000)

# tpg population size
parser.add_option('-p', '--pop', type='int', dest='popSize', default=200)

# whether or not to use memory
# either way tpg uses a form of memory on each learner in the form of registers
# this defines whether to use a long term form of memory shared between agents
parser.add_option('-m', '--mem', action='store_true', dest='useMem', default=False)

# number of classes to use for samples each generation
# if 0 all classes are used
# else the specified number of classes is used (randomly selected each generation)
parser.add_option('-c', '--nClasses', type='int', dest='nClasses', default=0)

# size of subsample to use
# if 0 full size is used
# else the specified size is used per each class (stratified)
parser.add_option('-s', '--ssize', type='int', dest='sampleSize', default=0)

# agent-wise or sample-wise execution order
# if true each agent guesses all samples then next agent goes
# else all agents guess a simple sample then the next sample is loaded
parser.add_option('-a', '--agentWise', action='store_true', dest='agentWise', default=False)

# fitness method to use
# if 0 use mean across classes
# if 1 use min across classes
# if 2 use static lexicographic
# if 3 use dynamic lexicographic
# if 4 use only current class score
# if 5 use only current class score + diminishing previous score
parser.add_option('-f', '--fitness', type='int', dest='fitnessMethod', default=0)

(options, args) = parser.parse_args()

################################################################################
# praparing cifar 10 data into dataframes
################################################################################

data = []
labels = []

# get all train data from all files into arrays
cifarDataPath = os.path.dirname(__file__) + "../cifar-10-batches-py/data_batch_"
for i in range(5):
    with open(cifarDataPath + str(i+1), "rb") as f:
        mdict = pickle.load(f, encoding="bytes")
        for i in range(len(mdict[b"data"])):
            data.append(mdict[b"data"][i])
            labels.append(mdict[b"labels"][i])

# put into dataframe for easy manipulation
trainDf = pd.DataFrame(labels).add_prefix("y").join(pd.DataFrame(data).add_prefix("x"))

# get test data
with open(os.path.dirname(__file__) +
                                "../cifar-10-batches-py/test_batch", "rb") as f:
    mdict = pickle.load(f, encoding="bytes")
    for i in range(len(mdict[b"data"])):
        data.append(mdict[b"data"][i])
        labels.append(mdict[b"labels"][i])

testDf = pd.DataFrame(labels).add_prefix("y").join(pd.DataFrame(data).add_prefix("x"))

################################################################################
# helper functions
################################################################################

# gets the data to be used in current gen
# number of samples from nClasses
def getDataForGeneration():
    # get proper sample size
    ss = options.sampleSize
    if ss == 0:
        ss = len(trainDf)

    # get some data from the desired classes
    # https://stackoverflow.com/a/44115314
    data = np.array(trainDf.groupby("y0", group_keys=False)
                        .apply(lambda x: x.sample(min(len(x), ss))))

    return data[:,1:], data[:,0]

# runs all agents on the specified data and labels
def runAgents(agents, data, labels):



################################################################################
# experiment setup
################################################################################

# set up tpg
trainer = Trainer(actions=range(10),
                  teamPopSize=options.popSize, rTeamPopSize=options.popSize)

# logs
# do it later (once stuff runs fine)

################################################################################
# execute experiment
################################################################################

for g in range(options.nGens):
