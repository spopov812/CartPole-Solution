# Overview-
Deep neural network trained to play the Cart Pole environment using Open AI's Gym (saved as a .h5 file). Trained used data achieving a score of 80 (out of 200 maximum). After training the model, the model achieves the highest possible score ~90% of the time.

## To run trained model provided-

$python Simulation.py

## To run and train new model-

$python Simulation.py train

## Note-
Open AI's Gym must be installed, as well as Keras library

After training, a graph will be generated showing the loss function change over time.
Only after closing it can the model procced with testing data.

During testing, the model will render the cartpole environment which will take additional time.
The window can be closed in order to speed evaluation of the model.
