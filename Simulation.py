from Model import build_model
from CartPoleTrain import get_training_data
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
import gym
import sys

train = False



if len(sys.argv) > 1 and sys.argv[1] == "train":
    train = True

env = gym.make("CartPole-v0")

x_data, y_data = get_training_data()

#builds and trains model
if train:
    model = build_model()
    history = model.fit(x_data, y_data, epochs=100)

    model.save('cartpolemodel.h5')

    # creates graph displaying loss rate
    plt.plot(history.history['mean_squared_error'])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

else:
    model = load_model('cartpolemodel.h5')

num_games_to_run = 100

scores = []

#runs games (100 by default)
for game in range(num_games_to_run):

    observation = env.reset()
    score = 0
    done = False

    #while a simulation has not been lost or max score achieved
    while not done:

        action = model.predict(observation.reshape(1, 4))

        if action >= .5:
            action = 1
        else:
            action = 0

        #env.render()

        #updated data after an action
        observation, reward, done, info = env.step(action)

        score += reward

    scores.append(score)

scores = np.array(scores)

print("Total score achieved after 100 simulations- ", scores.sum())

print("Max score achieved- ", np.max(scores))

print("Mean score- ", np.mean(scores))

print("Median score- ", np.median(scores))

not_max_score_vals = scores[scores != 200]
num_not_max_score_vals = len(not_max_score_vals)

print("The model failed to achieve the max score %d times, their values were:\n%s" % (num_not_max_score_vals, not_max_score_vals))
