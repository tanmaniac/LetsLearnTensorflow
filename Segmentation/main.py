import os
import glob
import zipfile
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers, losses, models
from tensorflow.python.keras import backend as K

import constants
import dataloader
import losses
import u_net

# Only import kaggle after loading credentials
import kaggle


def get_kaggle_credentials():
    token_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    token_file = os.path.join(token_dir, "kaggle.json")
    if not os.path.isdir(token_dir):
        os.mkdir(token_dir)
    try:
        with open(token_file, "r") as f:
            pass
    except IOError as no_file:
        try:
            from google.colab import files
        except ImportError:
            raise no_file

        uploaded = files.upload()

        if "kaggle.json" not in uploaded:
            raise ValueError("You need an API key! see: "
                             "https://github.com/Kaggle/kaggle-api#api-credentials")

        with open(token_file, "wb") as f:
            f.write(uploaded["kaggle.json"])
        os.chmod(token_file, 600)


def load_data_from_zip(competition, file):
    """
    Downlaod data from Kaggle and unzip files
    """
    with zipfile.ZipFile(os.path.join(competition, file), "r") as zip_ref:
        unzipped_file = zip_ref.namelist()[0]
        zip_ref.extractall(competition)


def get_data(competition):
    kaggle.api.competition_download_files(competition, competition)
    load_data_from_zip(competition, "train.zip")
    load_data_from_zip(competition, "train_masks.zip")
    load_data_from_zip(competition, "train_masks.csv.zip")


def visualize_samples(display_num, x_train_filenames, y_train_filenames):
    r_choices = np.random.choice(len(x_train_filenames), display_num)

    plt.figure(figsize=(10, 15))
    for i in range(0, display_num * 2, 2):
        img_num = r_choices[i // 2]
        x_pathname = x_train_filenames[img_num]
        y_pathname = y_train_filenames[img_num]

        plt.subplot(display_num, 2, i + 1)
        plt.imshow(mpimg.imread(x_pathname))
        plt.title("Original Image")

        example_labels = Image.open(y_pathname)
        label_vals = np.unique(example_labels)

        plt.subplot(display_num, 2, i + 2)
        plt.imshow(example_labels)
        plt.title("Masked Image")

    plt.suptitle("Examples of Images and their Masks")
    plt.show()


def main():
    # get_kaggle_credentials()
    # Download the data
    # get_data(constants.competition_name)

    img_dir = os.path.join(constants.competition_name, "train")
    label_dir = os.path.join(constants.competition_name, "train_masks")
    # Read label data
    df_train = pd.read_csv(os.path.join(
        constants.competition_name, "train_masks.csv"))
    ids_train = df_train["img"].map(lambda s: s.split(".")[0])

    x_train_filenames = []
    y_train_filenames = []
    for img_id in ids_train:
        x_train_filenames.append(os.path.join(
            img_dir, "{}.jpg".format(img_id)))
        y_train_filenames.append(os.path.join(
            label_dir, "{}_mask.gif".format(img_id)))

    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(
        x_train_filenames, y_train_filenames, test_size=0.2, random_state=42)

    num_train_examples = len(x_train_filenames)
    num_val_examples = len(x_val_filenames)

    print("Number of training examples: {}".format(num_train_examples))
    print("Number of validation examples: {}".format(num_val_examples))
    print("x_train_filenames: {}".format(x_train_filenames[:5]))
    print("y_train_filenames: {}".format(y_train_filenames[:5]))

    #visualize_samples(5, x_train_filenames, y_train_filenames)

    # Apply data augmentation to the training dataset - but NOT the validation one!
    tr_cfg = {
        "resize": [constants.img_shape[0], constants.img_shape[1]],
        "scale": 1 / 255.0,
        "hue_delta": 0.1,
        "horizontal_flip": True,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1
    }
    tr_preprocessing_fn = functools.partial(dataloader._augment, **tr_cfg)

    val_cfg = {
        "resize": [constants.img_shape[0], constants.img_shape[1]],
        "scale": 1 / 255.0
    }
    val_preprocessing_fn = functools.partial(dataloader._augment, **val_cfg)

    train_ds = dataloader.get_baseline_dataset(
        x_train_filenames, y_train_filenames, preproc_fn=tr_preprocessing_fn, batch_size=constants.batch_size)
    val_ds = dataloader.get_baseline_dataset(
        x_val_filenames, y_val_filenames, preproc_fn=val_preprocessing_fn, batch_size=constants.batch_size)

    # Set up model!

    model = u_net.unet(constants.img_shape)
    model.compile(optimizer="adam", loss=losses.bce_dice_loss,
                  metrics=[losses.dice_loss])
    model.summary()

    # Train!
    save_model_path = "/tmp/weights.hdf5"
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model_path, monitor="val_dice_loss", save_best_only=True, verbose=1)

    history = model.fit(train_ds, steps_per_epoch=int(np.ceil(num_train_examples / float(constants.batch_size))), epochs=constants.epochs,
                        validation_data=val_ds, validation_steps=int(np.ceil(num_val_examples / float(constants.batch_size))), callbacks=[cp])

    dice = history.history['dice_loss']

    val_dice = history.history['val_dice_loss']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(constants.epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='Training Dice Loss')
    plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Dice Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


if __name__ == "__main__":
    # Need this to train on RTX GPUs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

    # Train with fp16
    #dtype = "float16"
    # tf.keras.backend.set_floatx(dtype)
    # tf.keras.backend.set_epsilon(1e-4)

    main()
