import functools

import tensorflow as tf
from tensorflow import data
from tensorflow import contrib as tfcontrib

import constants


def _process_pathnames(fname, label_path):
    """
    Load an image and its label given its filename and path
    """
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    label_img_str = tf.read_file(label_path)
    # These are gif images so they return as (num_frames, h, w, c)
    label_img = tf.image.decode_gif(label_img_str)[0]
    # The label image is binary, so we just take the first channel
    label_img = label_img[:, :, 0]
    # But we still want it to be a 3-dimensional tensor instead of a 2d one
    label_img = tf.expand_dims(label_img, axis=-1)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """
    Shift by a random horizontal and/or vertical amount
    """
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform(
                [], -width_shift_range * constants.img_shape[1], width_shift_range * constants.img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform(
                [], -height_shift_range * constants.img_shape[0], height_shift_range * constants.img_shape[0])

        # Translate both
        output_img = tfcontrib.image.translate(
            output_img, [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(
            label_img, [width_shift_range, height_shift_range])

    return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(
                                        tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
    return tr_img, label_img


def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically
    if resize is not None:
        # Resize both images
        label_img = tf.image.resize_images(label_img, resize)
        img = tf.image.resize_images(img, resize)

    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)

    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(
        img, label_img, width_shift_range, height_shift_range)
    label_img = tf.to_float(label_img) * scale
    img = tf.to_float(img) * scale
    return img, label_img


def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5,
                         batch_size=constants.batch_size,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and "resize" not in preproc_fn.keywords:
        assert batch_size == 1, "Batched images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # We need to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset
