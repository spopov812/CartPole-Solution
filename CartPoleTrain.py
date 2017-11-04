import numpy as np
import gym


def get_training_data():

    env = gym.make('CartPole-v0')

    data_points = 512
    min_score = 50

    x_training_data = []
    y_training_data = []

    num_simulations = 0

    scores_from_simulations = []

    num_data_acquired = 0

    while num_data_acquired < data_points:

        num_simulations += 1

        total_score = 0

        x_training_sample = []
        y_training_sample = []

        done = False
        observation = env.reset()

        while not done:

            #env.render()

            action = np.random.randint(0, 2)

            x_training_sample.append(observation)
            y_training_sample.append(action)

            observation, reward, done, info = env.step(action)

            total_score += reward

        if total_score > min_score:

            scores_from_simulations.append(total_score)
            x_training_data += x_training_sample
            y_training_data += y_training_sample

            num_data_acquired += 1

            if num_data_acquired % 50 == 0:
                print("Acquired %d pairs of data" % num_data_acquired)

    x_training_data, y_training_data = np.array(x_training_data), np.array(y_training_data)

    print("\nNumber of games played- %d\n" % num_simulations)

    print("Average score in training data set is ", np.mean(scores_from_simulations))
    print("Median score in training data set is ", np.median(scores_from_simulations))

    return x_training_data, y_training_data
