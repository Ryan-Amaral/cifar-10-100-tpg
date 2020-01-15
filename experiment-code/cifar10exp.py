from tpg.trainer import Trainer
from tpg.agent import Agent
import pickle
import numpy as np
from optparse import OptionParser
import os
import pandas as pd
import random
import copy

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
# 1 to 10
parser.add_option('-c', '--nClasses', type='int', dest='nClasses', default=10)

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
parser.add_option('-f', '--fitness', type='int', dest='fitnessMethod', default=0)

(options, args) = parser.parse_args()

################################################################################
# praparing cifar 10 data into dataframes
################################################################################

data = []
labels = []

print("Loading all training data...")
# get all train data from all files into arrays
cifarDataPath = os.path.dirname(__file__) + "/../cifar-10-batches-py/data_batch_"
for i in range(5):
    with open(cifarDataPath + str(i+1), "rb") as f:
        mdict = pickle.load(f, encoding="bytes")
        for i in range(len(mdict[b"data"])):
            data.append(mdict[b"data"][i])
            labels.append(mdict[b"labels"][i])

print("Creating dataframe from training data...")
# put into dataframe for easy manipulation for shuffling and subsampling
trainDf = pd.DataFrame(labels[:40000]).add_prefix("y").join(pd.DataFrame(data[:40000]).add_prefix("x"))


# validation data to check after each generation
valData = np.array(data[40000:], dtype=float)
valLabels = np.array(labels[40000:], dtype=int)
valCounts = [0]*10
tmp = pd.Series(valLabels)
for c in range(10):
    valCounts[c] = (tmp.values == c).sum()



print("Loading test data...")
testData = []
testLabels = []
# get test data
with open(os.path.dirname(__file__) +
                                "/../cifar-10-batches-py/test_batch", "rb") as f:
    mdict = pickle.load(f, encoding="bytes")
    for i in range(len(mdict[b"data"])):
        testData.append(mdict[b"data"][i])
        testLabels.append(mdict[b"labels"][i])

testData = np.array(testData, dtype=float)
testLabels = np.array(testLabels, dtype=int)

testCounts = [0]*10
tmp = pd.Series(testLabels)
for c in range(10):
    testCounts[c] = (tmp.values == c).sum()

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

    # classes to select
    classes = random.sample(range(10), options.nClasses)
    # get up to ss amount of data from each class shuffled
    dataDf = (trainDf[trainDf["y0"].isin(classes)].groupby("y0", group_keys=False)
                        .apply(lambda x: x.sample(min(len(x), ss))).sample(frac=1))

    # get the actual count of data from each class
    counts = [0]*10
    for c in classes:
        counts[c] = (dataDf["y0"].values == c).sum()

    data = np.array(dataDf, dtype=float)

    return data[:,1:], np.array(data[:,0], dtype=int), counts


sampleSuccesses = []

# run one agent on all data (samples)
def runAgentWise(agent, samples, labels, counts):
    for i in range(len(samples)):
        guess = agent.act(samples[i]/255)
        score = guess == labels[i]
        sampleSuccesses[i] += score
        agent.team.outcomes[labels[i]] = (agent.team.outcomes.get(labels[i], 0) +
                                          score/counts[labels[i]])

# run all agents on a single data sample
def runSampleWise(agents, sample, label, count, i):
    for agent in agents:
        guess = agent.act(sample/255)
        score = guess == label
        sampleSuccesses[i] += score
        agent.team.outcomes[label] = agent.team.outcomes.get(label, 0) + score/count

# runs all agents on the specified data and labels
def runAgents(agents, data, labels, counts):

    # reset scores to reobtain
    for agent in agents:
        for i in range(10):
            if counts[i] > 0:
                agent.team.outcomes[i] = 0

    # save successes on each sample
    sampleSuccesses = np.zeros(len(data))

    # same agent on all data
    if options.agentWise:
        for agent in agents:
            runAgentWise(agent, data, labels, counts)

    # cycle through all agents on each data
    else:
        for i in range(len(data)):
            runSampleWise(agents, data[i], labels[i], counts[labels[i]], i)

    # normalize it
    sampleSuccesses /= len(data)

################################################################################
# experiment setup
################################################################################

print("Setting up TPG...")
# set up tpg
trainer = Trainer(actions=range(10),
                  teamPopSize=options.popSize, rTeamPopSize=options.popSize)

# logs
# do it later (once stuff runs fine)

# get the correct fitness method
if options.fitnessMethod == 0:
    fitnessType = "average"
elif options.fitnessMethod == 1:
    fitnessType = "min"
elif options.fitnessMethod == 2:
    fitnessType = "staticLexicographic"
elif options.fitnessMethod == 3:
    fitnessType = "dynamicLexicographic"


################################################################################
# execute experiment
################################################################################

for g in range(options.nGens):
    print("Starting Generation #" + str(g+1) + ".")
    print("Getting data for current generation...")
    # obtain new data every 5 generations
    if g % 5 == 0:
        data, labels, counts = getDataForGeneration()
    agents = trainer.getAgents(skipTasks=[c for c in range(10) if counts[c] > 0])
    print("Running agents...")
    runAgents(agents, data, labels, counts)

    print("\nReporting generational results...")

    print("Overall best agent (on train set):")
    bestAgent = trainer.getAgents(sortTasks=[c for c in range(10) if counts[c] > 0],
                                  multiTaskType=fitnessType)[0]
    print(bestAgent.team.outcomes)

    print("Evolving...")
    trainer.evolve(tasks=[c for c in range(10) if counts[c] > 0], multiTaskType=fitnessType)

    print("Overall best agent (on validation set):")
    originalOutcomes = copy.deepcopy(bestAgent.team.outcomes)
    bestAgent.team.outcomes = {}
    runAgentWise(bestAgent, valData, valLabels, valCounts)
    print(bestAgent.team.outcomes)
    bestAgent.team.outcomes = originalOutcomes
