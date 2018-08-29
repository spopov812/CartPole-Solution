from SupervisedLearning.Model import build_model
from SupervisedLearning.CartPoleTrain import get_training_data
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
import gym
import os


'''
Class that implements methods needed to solve cartpole environment using supervised learning.
'''
class SL:

    # constructor
    def __init__(self):

        # creates gym environment
        self.env = gym.make("CartPole-v0")

        # builds untrained neural network
        self.model = build_model()

        # number of games to run for testing
        self.num_games_to_run = 100

    # builds and trains model
    def train(self):

        # obtains training data that will be used to train model by performing random moves
        x_data, y_data = get_training_data()

        # trains model on data
        history = self.model.fit(x_data, y_data, epochs=10)

        # saves model
        self.model.save('SupervisedLearning/cartpolemodel.h5')

        # creates graph displaying loss rate
        plt.plot(history.history['mean_squared_error'])
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()

    # tests the performance of the model in the environmnet
    def test(self, modelname=''):

        # if user wants to load a model
        if modelname != '':
            self.model = load_model('SupervisedLearning/' + modelname)

        scores = []
        os.system("clear")

        # runs games
        for game in range(self.num_games_to_run):

            observation = self.env.reset()
            score = 0
            done = False

            # while a simulation has not been lost or max score achieved
            while not done:

                # asks model to make an action given a state
                action = self.model.predict(observation.reshape(1, 4))

                # decodes action
                if action >= .5:
                    action = 1
                else:
                    action = 0

                self.env.render()

                # updated data after an action
                observation, reward, done, info = self.env.step(action)

                score += reward

            scores.append(score)

        scores = np.array(scores)

        # prints statistics about testing
        self._print_statistics(scores)

    '''
    Prints statistics after testing a model.
    '''
    def _print_statistics(self, scores):

        print("\nTotal score achieved after %d simulations- %d" % (self.num_games_to_run, scores.sum()))
        print("Max score- %d" % np.max(scores))
        print("Mean score- %.2f" % np.mean(scores))
        print("Median score- %d" % np.median(scores))
