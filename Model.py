from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_model():

    #dense model with 3 hidden layers
    model = Sequential()

    model.add(Dense(24, input_shape=(4,), activation="relu"))
    model.add(Dropout(.6))

    model.add(Dense(24, activation="relu"))
    model.add(Dropout(.6))

    model.add(Dense(24, activation="relu"))
    model.add(Dropout(.6))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(

        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_squared_error']

        )

    return model
