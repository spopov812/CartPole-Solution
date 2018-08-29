# Overview-
Deep neural networks trained to play the Cart Pole environment using Open AI's Gym.
Two solutions are presented using both supervised and reinforcement learning (Deep Q learning).

# To run

## Specify one agent

For supervised learning add a "sl" tag before running the script
$ python Runme.py sl

For reinforcement learning (Q Learning) add a "rl" tag before running the script
$ python Runme.py rl

## To run or train new model-

$ python .py train
$ python Runme.py test

## Note-
Command line args for running or training can be combined such as
$ python Runme.py train test sl
$ python Runme.py test rl

Open AI's Gym must be installed, as well as Keras on top of tensorflow

After training a supervised learning model, a graph will be generated showing the loss function change over time.
Only after closing it can the model proceed with testing data.

During testing, the model may render the cartpole environment which will take additional time.
The window can be closed in order to speed evaluation of the model.
