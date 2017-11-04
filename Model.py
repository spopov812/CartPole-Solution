from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_model():

    model = Sequential()

    model.add(Dense(3, input_shape=(4,), activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(3, activation="relu"))
    model.add(Dropout(.5))

    model.add(Dense(2, activation="softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
        )

    return model
