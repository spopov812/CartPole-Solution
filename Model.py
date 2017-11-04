from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_model():

    model = Sequential()

    model.add(Dense(4, input_shape=(4,), activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(1, activation="softmax"))

    model.compile(

        loss="mean_squared_error",
        optimizer="adam",
        metrics=["accuracy"]
        )

    return model
