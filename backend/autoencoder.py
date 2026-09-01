"""Autoencoder from Colorization_of_BW_pictures.ipynb: Lab L -> ab."""

from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
from tensorflow.keras.models import Model


def build_colorizer(input_size: int = 128) -> Model:
    encoder_input = Input(shape=(input_size, input_size, 1))
    encoder_output = Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3, 3), activation="relu", padding="same")(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation="relu", padding="same")(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation="relu", padding="same", strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation="relu", padding="same")(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation="relu", padding="same")(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation="relu", padding="same")(encoder_output)

    decoder_output = Conv2D(128, (3, 3), activation="relu", padding="same")(encoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation="relu", padding="same")(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation="relu", padding="same")(decoder_output)
    decoder_output = Conv2D(16, (3, 3), activation="relu", padding="same")(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation="tanh", padding="same")(decoder_output)

    return Model(inputs=encoder_input, outputs=decoder_output)
