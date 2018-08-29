import sys
from QLearning.Simulation import RLAgent
from SupervisedLearning.Simulation import SL

agent = None
modelname = ''

if 'rl' in sys.argv:
    agent = RLAgent()
    modelname = "QLearning/"

elif 'sl' in sys.argv:
    agent = SL()
    modelname = "SupervisedLearning/"

else:
    print("No agent specified")
    exit()


if 'train' in sys.argv:
    agent.train()

if 'test' in sys.argv:

    modelname = ''

    if 'train' not in sys.argv:

        modelname += input("What is the name of the model- ")

    agent.test(modelname)
