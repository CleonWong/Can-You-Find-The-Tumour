from logs import logDecorator as lD
import jsonref

# For data loading:
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For data augmentation:
import random

# For weighted pixelwise crossentropy
from tensorflow.keras import backend as K

# For model building:
import tensorflow as tf
from tensorflow import keras

# For saving model:
import pickle

# For saving model params:
import json

# Miscellaneous:
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime as dt

# ---------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".lib.unet.unetVGG16"
config_unet = jsonref.load(open("../config/modules/unet.json"))

# ---------------------


class unetVgg16:
    def __init__(self, config_unet=config_unet):

        """
        This is the __init__ function of Class unet.

        This __init__ function is automatically called when a new instance is
        called. It calles the config file "../condig/modules/unet.json" and sets
        up multiple variables for the unet model.

        Parameters
        ----------
        config_unet : {json file}
            The json file for config_unet.

        Returns
        -------
        {}: {}
        """

        # Seeding.
        self.RSEED = config_unet["seed"]

        # Used in self.buildEncoder():
        self.encoder_input_width = config_unet["model"]["encoder"]["input_width"]
        self.encoder_input_channels = config_unet["model"]["encoder"]["input_channels"]
        self.encoder_input_shape = (
            self.encoder_input_width,
            self.encoder_input_width,
            self.encoder_input_channels,
        )

        # Used in self.buildUnet():
        self.kernsize = config_unet["model"]["decoder"]["kernel_size"]
        self.decoder_kernel_size = (self.kernsize, self.kernsize)
        self.stride = config_unet["model"]["decoder"]["strides"]
        self.decoder_strides = (self.stride, self.stride)
        self.decoder_padding = config_unet["model"]["decoder"]["padding"]
        self.decoder_activation = config_unet["model"]["decoder"]["activation"]
        self.final_layer_filters = config_unet["model"]["final_layer"]["filters"]
        self.final_layer_activation = config_unet["model"]["final_layer"]["activation"]

        # Used in self.datasetPaths():
        self.train_full_img_dir = config_unet["datasetPaths"]["train_full_img_dir"]
        self.train_mask_img_dir = config_unet["datasetPaths"]["train_mask_img_dir"]
        self.test_full_img_dir = config_unet["datasetPaths"]["test_full_img_dir"]
        self.test_mask_img_dir = config_unet["datasetPaths"]["test_mask_img_dir"]
        self.extension = config_unet["datasetPaths"]["extension"]

        # Used in load_image():
        self.target_size = (
            config_unet["load_image"]["target_size"],
            config_unet["load_image"]["target_size"],
        )

        # Used in imgAugment():
        self.brightness_delta = config_unet["imgAugment"]["brightness_delta"]

        # Used in makeTFDataset():
        self.batch_size = config_unet["makeTFDataset"]["batch_size"]

        # Used in self.train():
        self.validate = config_unet["train"]["validate"]
        self.loss = config_unet["train"]["loss"]
        self.learning_rate = config_unet["train"]["learning_rate"]
        self.dropout = config_unet["train"]["dropout"]
        self.dropout_training = config_unet["train"]["dropout_training"]
        self.num_epochs = config_unet["train"]["num_epochs"]
        self.callback_monitor = config_unet["train"]["callback_monitor"]
        self.callback_mode = config_unet["train"]["callback_mode"]
        self.ckpt_save_weights_only = config_unet["train"]["ckpt_save_weights_only"]
        self.ckpt_save_best_only = config_unet["train"]["ckpt_save_best_only"]
        self.earlystop_patience = config_unet["train"]["earlystop_patience"]
        self.restore_best_weights = config_unet["train"][
            "earlystop_restore_best_weights"
        ]
        self.results_dir = config_unet["train"]["results_dir"]

    @lD.log(logBase + ".buildEncoder")
    def buildEncoder(logger, self):

        """
        =========
         Encoder
        =========

        Creates the encoder part of the unet by importing pretrained VGG16 from
        Keras API.

        Parameters
        ----------
        {}: {}

        Returns
        -------
        encoder_model: {}
            The encoder part of the Unet (VGG16) with frozen weights pretrained
            from ImageNet dataset and last fully connected layer not included.
        """

        try:
            # Get base model
            VGG16_ = keras.applications.VGG16(
                include_top=False,
                weights="imagenet",
                input_shape=self.encoder_input_shape,
            )

            # Get list of layer names for skip connections later
            layer_names = [layer.name for layer in VGG16_.layers]

            # Get layer outputs
            all_layer_outputs = [
                VGG16_.get_layer(layer_name).output for layer_name in layer_names
            ]

            # Create encoder model
            encoder_model = keras.Model(inputs=VGG16_.input, outputs=all_layer_outputs)

            # Freeze layers
            encoder_model.trainable = False

        except Exception as e:
            # logger.error(f'Unable to buildEncoder!\n{e}')
            print((f"Unable to buildEncoder!\n{e}"))

        return encoder_model

    @lD.log(logBase + ".buildUnet")
    def buildUnet(logger, self, dropout_training):

        """
        ============
         Unet_VGG16
        ============

        Creates the unet (VGG16) based on the following flow:
            - unet_input
                  |
              encoder_model (block)
                  |
              encoded_img (output)
                  |
              decoder_model (block)
                  |
              decoded_img (output)
                  |
              final_up_conv (layer)
                  |
              final_img (output)

            1. Creates input layer (`unet_input`).
                - Uses: Input()
                - Output: `unet_input`
            2. Creates encoder block.
                - Input: `unet_input`
                - Uses: self.buildEncoder()
                - Output: `encoded_img`
            3. Creates decoder block.
                - Input: `encoded_img`
                - Uses: Conv2D(), Conv2DTranspose(), Concatenate()
                - Output: `decoded_img`
            4. Creates final conv layer.
                - Input: `decoded_img`
                - Uses: Conv2D()
                - Output: `final_img`

        The decoder part of the model is the reverse of the VGG16 plus dropout
        layers betwee each block.

        Parameters
        ----------
        dropout_training: {boolean}
            If `True`, dropout layers will be activated (should be True during
            training).
            If `False`, dropout layers will NOT be activated (should be False
            during prediction).

        Returns
        -------
        unet: {}
            The unet model.
        """

        try:
            # =============
            #  Input layer
            # =============

            unet_input = keras.Input(
                shape=self.encoder_input_shape, name="unet_input_layer"
            )

            x = unet_input

            # =========
            #  Encoder
            # =========

            encoder_model = self.buildEncoder()
            all_encoder_layer_outputs = encoder_model(x)

            # Get final encoder output (this will be the input for the decoder)
            encoded_img = all_encoder_layer_outputs[-1]

            # Get outputs to be used for skip connections
            # (I know the specific layers to be used for skip connections)
            skip_outputs = [all_encoder_layer_outputs[i] for i in [2, 5, 9, 13, 17]]

            # =========
            #  Decoder
            # =========

            decoder_filters = int(encoded_img.shape[-1])

            # ------------------------------------------
            # Block 5: 7x7 -> 14x14
            #  - `encoded_img` as initial input for decoder
            x = keras.layers.Conv2DTranspose(
                name="block5_up_convT",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=self.decoder_strides,
                padding=self.decoder_padding,
                activation=self.decoder_activation,
            )(encoded_img)

            x = keras.layers.Concatenate(name="block5_up_concat", axis=-1)(
                [x, skip_outputs[4]]
            )

            x = keras.layers.Dropout(
                name="block5_up_dropout", rate=self.dropout, seed=self.RSEED
            )(x, training=dropout_training)

            x = keras.layers.Conv2D(
                name="block5_up_conv3",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block5_up_conv2",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block5_up_conv1",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)

            # ------------------------------------------
            # Block 4: 14x14 -> 28x28
            x = keras.layers.Conv2DTranspose(
                name="block4_up_convT",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=self.decoder_strides,
                padding=self.decoder_padding,
                activation=self.decoder_activation,
            )(x)

            x = keras.layers.Concatenate(name="block4_up_concat", axis=-1)(
                [x, skip_outputs[3]]
            )

            x = keras.layers.Dropout(
                name="block4_up_dropout", rate=self.dropout, seed=self.RSEED
            )(x, training=dropout_training)

            x = keras.layers.Conv2D(
                name="block4_up_conv3",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block4_up_conv2",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block4_up_conv1",
                filters=decoder_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)

            # ------------------------------------------
            # Block 3: 28x28 -> 56x56
            x = keras.layers.Conv2DTranspose(
                name="block3_up_convT",
                filters=int(decoder_filters / 2),
                kernel_size=self.decoder_kernel_size,
                strides=self.decoder_strides,
                padding=self.decoder_padding,
                activation=self.decoder_activation,
            )(x)

            x = keras.layers.Concatenate(name="block3_up_concat", axis=-1)(
                [x, skip_outputs[2]]
            )

            x = keras.layers.Dropout(
                name="block3_up_dropout", rate=self.dropout, seed=self.RSEED
            )(x, training=dropout_training)

            x = keras.layers.Conv2D(
                name="block3_up_conv3",
                filters=int(decoder_filters / 2),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block3_up_conv2",
                filters=int(decoder_filters / 2),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block3_up_conv1",
                filters=int(decoder_filters / 2),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)

            # ------------------------------------------
            # Block 2: 56x56 -> 112x112
            x = keras.layers.Conv2DTranspose(
                name="block2_up_convT",
                filters=int(decoder_filters / 4),
                kernel_size=self.decoder_kernel_size,
                strides=self.decoder_strides,
                padding=self.decoder_padding,
                activation=self.decoder_activation,
            )(x)

            x = keras.layers.Concatenate(name="block2_up_concat", axis=-1)(
                [x, skip_outputs[1]]
            )

            x = keras.layers.Dropout(
                name="block2_up_dropout", rate=self.dropout, seed=self.RSEED
            )(x, training=dropout_training)

            x = keras.layers.Conv2D(
                name="block2_up_conv2",
                filters=int(decoder_filters / 4),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            x = keras.layers.Conv2D(
                name="block2_up_conv1",
                filters=int(decoder_filters / 4),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)

            # ------------------------------------------
            # Block 1: 112x112 -> 224x224
            x = keras.layers.Conv2DTranspose(
                name="block1_up_convT",
                filters=int(decoder_filters / 8),
                kernel_size=self.decoder_kernel_size,
                strides=self.decoder_strides,
                padding=self.decoder_padding,
                activation=self.decoder_activation,
            )(x)

            x = keras.layers.Concatenate(name="block1_up_concat", axis=-1)(
                [x, skip_outputs[0]]
            )

            x = keras.layers.Dropout(
                name="block1_up_dropout", rate=self.dropout, seed=self.RSEED
            )(x, training=dropout_training)

            x = keras.layers.Conv2D(
                name="block1_up_conv2",
                filters=int(decoder_filters / 8),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)
            decoded_img = keras.layers.Conv2D(
                name="block1_up_conv1",
                filters=int(decoder_filters / 8),
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )(x)

            # ------------------------------------------
            # Final conv layer
            final_img = keras.layers.Conv2D(
                name="final_up_conv",
                filters=self.final_layer_filters,
                kernel_size=self.decoder_kernel_size,
                strides=(1, 1),
                padding="same",
                activation=self.final_layer_activation,
            )(decoded_img)

            # ======
            #  Unet
            # ======

            unet = keras.Model(inputs=unet_input, outputs=final_img, name="Unet_VGG16")

        except Exception as e:
            # logger.error(f'Unable to buildUnet!\n{e}')
            print((f"Unable to buildUnet!\n{e}"))

        return unet

    @lD.log(logBase + ".datasetPaths")
    def datasetPaths(
        logger,
        self,
        full_img_dir,
        mask_img_dir,
        extension,
    ):

        """
        Follows the flow:
        1.         datasetPaths() *
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the directories of the folder containing the full mammogram
        scans (x) and the ground truth masks (y) and returns a list of paths to
        individual full scans and masks.

        Parameters
        ----------
        full_img_dir : {str}
            Directory that contains the FULL training images.
        mask_img_dir : {str}
            Directory that contains the MASK training images.
        extension : {str}
            The file extension of the images (e.g. ".png")

        Returns
        -------
        x_paths_list: {list}
            A list of paths to individual FULL images in the training set.
        y_paths_list: {list}
            A list of paths to individual MASK images in the training set.
        """

        try:

            # =======================================
            #  1. Get paths of X (full) and y (mask)
            # =======================================

            x_paths_list = []
            y_paths_list = []

            # Get paths of train images and masks.
            for full in os.listdir(full_img_dir):
                if full.endswith(extension):
                    x_paths_list.append(os.path.join(full_img_dir, full))

            for mask in os.listdir(mask_img_dir):
                if mask.endswith(extension):
                    y_paths_list.append(os.path.join(mask_img_dir, mask))

            # ** IMPORTANT ** Sort so that FULL and MASK images are in an order
            # that corresponds with each other.
            x_paths_list.sort()
            y_paths_list.sort()

        except Exception as e:
            # logger.error(f'Unable to datasetPaths!\n{e}')
            print((f"Unable to datasetPaths!\n{e}"))

        return x_paths_list, y_paths_list

    @lD.log(logBase + ".loadFullImg")
    def loadFullImg(logger, self, path, dsize):

        """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()* loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the path of a FULL image and loads it as a grayscale image (as
        a numpy array with 3 channels).

        Parameters
        ----------
        path : {str}
            The path of the FULL image.

        Returns
        -------
        full_img : {numpy.ndarray}
            The loaded image with shape = (self.target_size, self.target_size, 3)
        """

        try:
            # =====================================
            #  2a. Read images (full)
            # =====================================
            if not isinstance(path, str):
                path = path.decode()

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(src=img, dsize=dsize)

            # Min max normalise to [0, 1].
            norm_img = (img - img.min()) / (img.max() - img.min())

            # Stack grayscale image to make channels=3.
            full_img = np.stack([norm_img, norm_img, norm_img], axis=-1)

        except Exception as e:
            # logger.error(f'Unable to loadFullImg!\n{e}')
            print((f"Unable to loadFullImg!\n{e}"))

        return full_img

    @lD.log(logBase + ".loadMaskImg")
    def loadMaskImg(logger, self, path, dsize):

        """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg() *
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the path of a MASK image and loads it as a grayscale image (as
        a numpy array with 1 channel).

        Parameters
        ----------
        path : {str}
            The path of the MASK image.

        Returns
        -------
        full_img : {numpy.ndarray}
            The loaded image with shape = (self.target_size, self.target_size, 1)
        """

        try:

            # ========================
            #  2b. Read images (mask)
            # ========================
            if not isinstance(path, str):
                path = path.decode()

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(src=img, dsize=dsize)

            # Min max normalise to [0, 1].
            norm_img = (img - img.min()) / (img.max() - img.min())

            # Expand shape to (width, height, 1).
            mask_img = np.expand_dims(norm_img, axis=-1)

        except Exception as e:
            # logger.error(f'Unable to loadMaskImg!\n{e}')
            print((f"Unable to loadMaskImg!\n{e}"))

        return mask_img

    @lD.log(logBase + ".tfParse")
    def tfParse(logger, self, x_path, y_path):

        """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg()
                |_______________|
                        |
        3.          tfParse() *
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()


        Takes in a FULL image path and a MASK image path, loads the images using
        loadFullImg() and loadMaskImg() and makes it TensorFlow executable.

        Parameters
        ----------
        x_path: {str}
            The path of the FULL image to be loaded.
        y_path: {str}
            The path of the MASK image to be loaded.

        Returns
        -------
        x : {tensor}
            The FULL image loaded as a tensor with shape =
            (self.target_size, self.target_size, 3)
        y : {tensor}
            The MASK image loaded as a tensor with shape =
            (self.target_size, self.target_size, 1)
        """

        try:
            # ===========
            #  3. Parse
            # ===========
            def _parse(x_path, y_path):
                x = self.loadFullImg(path=x_path, dsize=self.target_size)
                y = self.loadMaskImg(path=y_path, dsize=self.target_size)
                return x, y

            x, y = tf.numpy_function(_parse, [x_path, y_path], [tf.float64, tf.float64])

            x.set_shape([self.target_size[0], self.target_size[0], 3])
            y.set_shape([self.target_size[0], self.target_size[0], 1])

            return x, y

        except Exception as e:
            # logger.error(f'Unable to tfParse!\n{e}')
            print((f"Unable to tfParse!\n{e}"))

    @lD.log(logBase + ".imgAugment")
    def imgAugment(logger, self, x_img, y_img):

        """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment() *
                        |
        5.        makeTFDataset()

        Apply random image augmentation to full mammogram scans (x_img). Note
        that the same augmentation has to be applied to the masks (y_img), apart
        from the brightness augmentation.

        Parameters
        ----------
        x_img : {numpy.ndarray}
            Full mammogram scan to augment.
        y_img : {numpy.ndarray}
            Corresponding mask of `x_img`.

        Returns
        -------
        x_img : {numpy.ndarray}
            Augmented x_img.
        y_img : {numpy.ndarray}
            Augmented y_img.
        """

        try:

            # =========
            #  LR Flip
            # =========
            # Generate random number from uniform distribution
            # in the range [0.0, 1.0)
            if tf.random.uniform(()) > 0.5:
                x_img = tf.image.flip_left_right(image=x_img)
                y_img = tf.image.flip_left_right(image=y_img)

            # =========
            #  UD Flip
            # =========
            if tf.random.uniform(()) > 0.5:
                x_img = tf.image.flip_up_down(image=x_img)
                y_img = tf.image.flip_up_down(image=y_img)

            # ============
            #  Brightness
            # ============
            # Only change the brightness of x_img, not y_img!
            x_img = tf.image.random_brightness(
                image=x_img, max_delta=self.brightness_delta
            )

        except Exception as e:
            # logger.error(f'Unable to imgAugment!\n{e}')
            print((f"Unable to imgAugment!\n{e}"))

        return x_img, y_img

    @lD.log(logBase + ".makeTFDataset")
    def makeTFDataset(
        logger, self, shuffle, augment, x_paths_list, y_paths_list, batch_size
    ):

        """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset() *


        Takes in a FULL image path and a MASK image path, loads the images using
        loadFullImg() and loadMaskImg() and makes it TensorFlow executable.

        Parameters
        ----------
        shuffle : {boolean}
            If True, shuffle dataset. Should only shuffle train set.
        augment : {boolean}
            If True, perform augmentation. Should only augment test set.
        x_paths_list : {list}
            The path of the FULL image to be loaded.
        y_paths_list : {list}
            The path of the MASK image to be loaded.
        batch_size : {int}
            The batch size to create the dataset with.

        Returns
        -------
        ds : {RepeatDataset}
            The dataset with x (FULL) and y (MASK) images loaded from
            `x_paths_list` and `y_paths_list`.

            For each `batch` in `ds`:
            - type = tuple
            - len = 2

            - `batch[0]` (this is a batch of the FULL images):
                - type = tensorflow.python.fraework.ops.EagerTensor
                - shape = (self.batch_size, self.target_size, self.target_size, 3)

                - `batch[0][0]` (this is the first FULL image in the batch):
                    - type = tensorflow.python.fraework.ops.EagerTensor
                    - shape = (self.target_size, self.target_size, 3)


            - `batch[1]` (this is a batch the MASK images):
                - type = tensorflow.python.fraework.ops.EagerTensor
                - shape = (self.batch_size, self.target_size, self.target_size, 1)

                    - `batch[1][0]` (this is the first MASK image in the batch):
                    - type = tensorflow.python.fraework.ops.EagerTensor
                    - shape = (self.target_size, self.target_size, 1)
        """

        try:
            # ====================
            #  4. Make TF Dataset
            # ====================

            ds = tf.data.Dataset.from_tensor_slices((x_paths_list, y_paths_list))

            # Shuffle the paths of the elements
            # with buffer_size=len(x_paths_list), since storing all the paths
            # in memory will not be an issue (as compared to storing all the
            # imported images as tensors).
            if shuffle:
                ds = ds.shuffle(buffer_size=len(x_paths_list))

            # Transform paths to images after shuffling the paths.
            ds = ds.map(self.tfParse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # Apply image augmentation.
            if augment:
                ds = ds.map(
                    self.imgAugment, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

            # Batch only AFTER shuffle, so that we shuffled the elements not the batches.
            ds = ds.batch(batch_size=batch_size, drop_remainder=False)
            ds = ds.repeat()  # We need to repeat in order to train for > 1 epoch.

        except Exception as e:
            # logger.error(f'Unable to makeTFDataset!\n{e}')
            print((f"Unable to makeTFDataset!\n{e}"))

        return ds

    @lD.log(logBase + ".iouMetric")
    def iouMetric(logger, self, y_true, y_pred):

        """
        Calculates the intersection-over-union between a ground truth mask and
        the corresponding predicted mask from the model.

        Parameters
        ----------
        y_true : {tf.Tensor}
            The ground truth mask.
        y_pred : {tf.Tensor}
            Corresponding predicted mask.

        Returns
        -------
        compute_iou() : {tf.numpy_function}
            A function that calculates the IOU metric.
        """

        try:

            def compute_iou(y_true, y_pred):
                intersection = (y_true * y_pred).sum()
                union = y_true.sum() + y_pred.sum() - intersection
                x = (intersection + 1e-15) / (union + 1e-15)
                x = x.astype(np.float32)
                return x

            return tf.numpy_function(compute_iou, [y_true, y_pred], tf.float32)

        except Exception as e:
            # logger.error(f'Unable to iouMetric!\n{e}')
            print((f"Unable to iouMetric!\n{e}"))

    @lD.log(logBase + ".compile_")
    def compile_(logger, self, model):

        """
        Compiles the model after building it.

        Parameters
        ----------
        model : {tf.Function}
            The unet that self.buildUnet() returns.

        Returns
        -------
        model : {tf.Function}
            The compiled model.
        """

        try:
            loss = keras.losses.BinaryCrossentropy(from_logits=False)
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            metrics = ["accuracy", self.iouMetric]
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        except Exception as e:
            # logger.error(f'Unable to compile_!\n{e}')
            print((f"Unable to compile_!\n{e}"))

        return model

    @lD.log(logBase + ".train")
    def train(logger, self):

        """
        Creates the model using self.buildUnet(), compiles the model, then
        trains the model.

        Parameters
        ----------
        {}: {}

        Returns
        -------
        {}: {}
        """

        try:

            # =====================================
            #  Create folder for this training run
            # =====================================
            model_time = dt.now().strftime("%Y%m%d_%H%M%S")
            model_folder = os.path.join(self.results_dir, model_time)

            # Parent folder
            os.makedirs(model_folder)

            # TensorBoard folder
            tensorboard_folder = os.path.join(model_folder, "tensorlogs")
            os.makedirs(tensorboard_folder)

            # Checkpoint folder
            ckpt_folder = os.path.join(model_folder, "checkpoints")
            os.makedirs(ckpt_folder)

            # CSV Logger folder
            csv_logger_folder = os.path.join(model_folder, "csv_logger")
            os.makedirs(csv_logger_folder)

            # History folder
            hist_folder = os.path.join(model_folder, "model_history")
            os.makedirs(hist_folder)

            # Saved model folder
            saved_model_folder = os.path.join(model_folder, "saved_model")
            os.makedirs(saved_model_folder)

            # Json params folder
            model_params_folder = os.path.join(model_folder, "model_params")
            os.makedirs(model_params_folder)

            # ===============
            #  Build dataset
            # ===============

            # Get train and test paths.
            train_x, train_y = self.datasetPaths(
                full_img_dir=self.train_full_img_dir,
                mask_img_dir=self.train_mask_img_dir,
                extension=self.extension,
            )

            test_x, test_y = self.datasetPaths(
                full_img_dir=self.test_full_img_dir,
                mask_img_dir=self.test_mask_img_dir,
                extension=self.extension,
            )

            # Create train and test datasets.
            train_ds = self.makeTFDataset(
                shuffle=True,
                augment=True,
                x_paths_list=train_x,
                y_paths_list=train_y,
                batch_size=self.batch_size,
            )

            if self.validate:
                test_ds = self.makeTFDataset(
                    shuffle=False,
                    augment=False,
                    x_paths_list=test_x,
                    y_paths_list=test_y,
                    batch_size=self.batch_size,
                )

            # =============
            #  Build model
            # =============
            unet = self.buildUnet(dropout_training=self.dropout_training)

            # ==========
            #  Compile
            # ==========
            unet = self.compile_(model=unet)
            print(unet.summary())

            # ===========
            #  Callbacks
            # ===========

            # Checkpoint
            ckpt_path = (
                ckpt_folder
                + f"/{model_time}"
                + "_Epoch-{epoch:03d}"
                + "_IOU-{iouMetric:.8f}"
            )
            ckpt_callback = keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor=self.callback_monitor,
                mode=self.callback_mode,
                save_weights_only=self.ckpt_save_weights_only,
                save_best_only=self.ckpt_save_best_only,
                verbose=1,
            )

            # Early Stopping
            es_callback = keras.callbacks.EarlyStopping(
                patience=self.earlystop_patience,
                monitor=self.callback_monitor,
                mode=self.callback_mode,
                restore_best_weights=self.restore_best_weights,
            )

            # TensorBoard
            tb_callback = keras.callbacks.TensorBoard(
                log_dir=tensorboard_folder, histogram_freq=1, profile_batch=0
            )

            # CSV Logger
            csv_logger_path = os.path.join(csv_logger_folder, "csv_logger.csv")
            csv_logger = keras.callbacks.CSVLogger(
                filename=csv_logger_path, separator=",", append=True
            )

            # Putting them together
            callbacks = [ckpt_callback, es_callback, tb_callback, csv_logger]

            # =====
            #  Fit
            # =====

            # Calculate train and test steps per epoch.
            train_steps = len(train_x) // self.batch_size
            test_steps = len(test_x) // self.batch_size

            if len(train_x) % self.batch_size != 0:
                train_steps += 1
            if len(test_x) % self.batch_size != 0:
                test_steps += 1

            print()
            print(f"Size of training set = {len(train_x)}")
            print(f"Size of test set = {len(test_x)}")
            print(f"Number of epochs = {self.num_epochs}")
            print(f"Batch size = {self.batch_size}")
            print(f"Number of training steps per epoch = {train_steps}")
            print(f"Number of test steps per epoch = {test_steps}")
            print()

            # Fit!
            if self.validate:
                history = unet.fit(
                    train_ds,
                    validation_data=test_ds,
                    epochs=self.num_epochs,
                    steps_per_epoch=train_steps,
                    validation_steps=test_steps,
                    callbacks=callbacks,
                    verbose=1,
                )
            elif not self.validate:
                history = unet.fit(
                    train_ds,
                    epochs=self.num_epochs,
                    steps_per_epoch=train_steps,
                    callbacks=callbacks,
                    verbose=1,
                )

            # ==========================
            #  Save relevant model infos
            # ==========================

            # Save model
            saved_model_path = os.path.join(saved_model_folder, model_time)
            unet.save(saved_model_path)

            # Save model params
            model_params_path = os.path.join(model_params_folder, "model_params.json")
            with open(model_params_path, "w") as f:
                json.dump(config_unet, f)

        except Exception as e:
            # logger.error(f'Unable to train!\n{e}')
            print((f"Unable to train!\n{e}"))

        return
