from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_model():

    model = Sequential()

    model.add(Dense(4, input_shape=(4,), activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.4))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(

        loss='mean_squared_error',
        optimizer='sgd',
        metrics=['binary_accuracy', 'mean_squared_error']

        )

    return model
