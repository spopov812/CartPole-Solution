To run trained model provided-

$python Simulation.py


To run and train new model(over 100 epochs with data that achieves score of at least 80)-

$python Simulation.py train

Note-
Open AI's Gym must be installed, as well as Keras library

Highest score possible is 200

After training, a graph will be generated showing the loss function change over time.
Only after closing it can the model procced with testing data.

During testing, the model will render the cartpole environment which will take additional time.
The window can be closed in order to speed evaluation of the model.
