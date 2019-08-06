import tensorflow as tf
from tensorflow.python.keras import layers, models


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding="same")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(
        num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding="same")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding="same")(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    return decoder


def unet(img_shape):
    inputs = layers.Input(shape=img_shape)  # 256 x 256 x 3
    encoder0_pool, encoder0 = encoder_block(inputs, 32)  # 128 x 128 x 32
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)  # 64 x 64 x 64
    encoder2_pool, encoder2 = encoder_block(
        encoder1_pool, 128)  # 32 x 32 x 128
    encoder3_pool, encoder3 = encoder_block(
        encoder2_pool, 256)  # 16 x 16 x 256
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)  # 8 x 8 x 512
    center = conv_block(encoder4_pool, 1024)  # 8 x 8 x 1024
    decoder4 = decoder_block(center, encoder4, 512)  # 16 x 16
    decoder3 = decoder_block(decoder4, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model
