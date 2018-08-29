from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import Sequential


def build_model():

    model = Sequential()

    model.add(Dense(32, input_shape=(4,), activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(2, activation="linear"))

    model.compile(

        optimizer=Adam(lr=0.001),
        loss='mse'

    )

    return model

