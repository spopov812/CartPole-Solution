from Model import build_model
import numpy as np
import gym


env = gym.make("CartPole-v0")

read_data = np.load("TrainingData.csv.npz")

x_data = read_data['x_data']
y_data = read_data['y_data']


model = build_model()
model.fit(x_data, y_data, epochs=7)

num_games_to_run = 50

scores = []

for game in num_games_to_run:

    observation = env.reset()
    score = 0
    done = False

    while not done:

        action = np.argmax(model.predict(observation.reshape(1, 4)))

        observation, reward, done, info = env.step(action)

        score += reward

    scores.append(score)

scores = np.array(scores)

print("Max score achieved- ", np.max(scores))

print("Mean score- ", np.mean(scores))

print("Median score- ", np.median(scores))
