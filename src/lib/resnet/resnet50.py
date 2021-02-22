from logs import logDecorator as lD
import jsonref

# For data loading:
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For createDataset():
import random
import pandas as pd

# For buildResNet50():
import tensorflow as tf
from tensorflow import keras

# For saving model params:
import json

# For train():
from datetime import datetime as dt

# ---------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".lib.resnet50.resnet50"
config_resnet = jsonref.load(open("../config/modules/resnet50.json"))

# ---------------------


class resnet50:
    def __init__(self, config_resnet=config_resnet):

        """
        This is the __init__ function of Class resnet50.

        This __init__ function is automatically called when a new instance is
        called. It calles the config file "../condig/modules/resnet50.json" and sets
        up multiple variables for the resnet model.

        Parameters
        ----------
        config_resnet : {json file}
            The json file for config_resnet.

        Returns
        -------
        {}: {}
        """

        # Seeding.
        self.RSEED = config_resnet["seed"]

        # Used in self.train():
        self.which_model = config_resnet["which_model"]

        # Used in self.buildResNet50(), self.buildVGG16():
        self.resnet_input_width = config_resnet["buildResNet50"]["input_width"]
        self.resnet_input_height = config_resnet["buildResNet50"]["input_height"]
        self.resnet_input_channels = config_resnet["buildResNet50"]["input_channels"]
        self.resnet_input_shape = (
            self.resnet_input_width,
            self.resnet_input_height,
            self.resnet_input_channels,
        )
        self.resnet_FC_neurons = config_resnet["buildResNet50"]["FC_neurons"]
        self.resnet_FC_activation = config_resnet["buildResNet50"]["FC_activation"]
        self.resnet_output_neurons = config_resnet["buildResNet50"]["output_neurons"]
        self.resnet_output_activation = config_resnet["buildResNet50"][
            "output_activation"
        ]

        # Used in self.createDataset():
        self.createDataset_all_imgs_dir = config_resnet["createDataset"]["all_imgs_dir"]
        self.createDataset_extension = config_resnet["createDataset"]["extension"]
        self.createDataset_val_split = config_resnet["createDataset"]["val_split"]
        self.createDataset_brightness_range = config_resnet["createDataset"][
            "brightness_range"
        ]
        self.createDataset_batch_size = config_resnet["createDataset"]["batch_size"]

        # Used in self.compile_():
        self.compile_optimiser = config_resnet["compile_"]["optimiser"]
        self.compile_learning_rate = config_resnet["compile_"]["learning_rate"]
        self.compile_metrics = config_resnet["compile_"]["metrics"]
        self.compile_loss = config_resnet["compile_"]["loss"]

        # Used in self.createCallbacks():
        self.callback_monitor = config_resnet["createCallbacks"]["callback_monitor"]
        self.callback_mode = config_resnet["createCallbacks"]["callback_mode"]
        self.ckpt_save_weights_only = config_resnet["createCallbacks"][
            "ckpt_save_weights_only"
        ]
        self.ckpt_save_best_only = config_resnet["createCallbacks"][
            "ckpt_save_best_only"
        ]
        self.earlystop_patience = config_resnet["createCallbacks"]["earlystop_patience"]
        self.restore_best_weights = config_resnet["createCallbacks"][
            "earlystop_restore_best_weights"
        ]

        # Used in self.train():
        self.num_epochs = config_resnet["train"]["num_epochs"]
        self.results_dir = config_resnet["train"]["results_dir"]

    @lD.log(logBase + ".buildResNet50")
    def buildResNet50(logger, self):

        """
        Builds the ResNet 50 architecture by importing pretrained ResNet50 from
        Keras API, and then adding FCC layers according to the needs of the
        classification task using the Keras Functional API method.

        Parameters
        ----------
        {}: {}

        Returns
        -------
        resnet50_model: {}
            The ResNet50 model imported from Keras API with frozen weights
            pretrained from ImageNet dataset with the last fully connected layer
            customised to our classification needs.
        """

        try:
            # Download ResNet50 architecture with ImageNet weights.
            base_model = keras.applications.ResNet50(
                include_top=False,
                weights="imagenet",
                input_shape=self.resnet_input_shape,
            )

            # Take the output of the last convolution block in ResNet50.
            x = base_model.output

            # Add Global Average Pooling layer.
            x = keras.layers.GlobalAveragePooling2D()(x)

            # Add FC layer having 1024 neurons.
            x = keras.layers.Dense(
                units=self.resnet_FC_neurons, activation=self.resnet_FC_activation
            )(x)

            # Add FC output layer for final classification.
            final_x = keras.layers.Dense(
                units=self.resnet_output_neurons,
                activation=self.resnet_output_activation,
            )(x)

            # Create ResNet50 model.
            resnet50_model = keras.Model(inputs=base_model.input, outputs=final_x)

            # Freeze layers of base model.
            for layer in base_model.layers:
                layer.trainable = False

        except Exception as e:
            # logger.error(f'Unable to buildResNet50!\n{e}')
            print((f"Unable to buildResNet50!\n{e}"))

        return resnet50_model

    @lD.log(logBase + ".builVGG16")
    def buildVGG16(logger, self):

        """
        Builds the VGG16 architecture by importing pretrained VGG16 from
        Keras API, and then adding fully connected layers according to the needs
        of the classification task using the Keras Functional API method.

        Parameters
        ----------
        {}: {}

        Returns
        -------
        VGG16_model: {}
            The VGG16 model imported from Keras API with frozen weights
            pretrained from ImageNet dataset with the last fully connected layer
            customised to our classification needs.
        """

        try:
            # Download VGG16 architecture with ImageNet weights.
            base_model = keras.applications.VGG16(
                include_top=False,
                weights="imagenet",
                input_shape=self.resnet_input_shape,
            )

            # Take the output of the last convolution block in VGG16.
            x = base_model.output

            # Add Global Average Pooling layer.
            x = keras.layers.GlobalAveragePooling2D()(x)

            # Add FC layer having 1024 neurons.
            x = keras.layers.Dense(
                units=self.resnet_FC_neurons, activation=self.resnet_FC_activation
            )(x)

            # Add FC output layer for final classification.
            final_x = keras.layers.Dense(
                units=self.resnet_output_neurons,
                activation=self.resnet_output_activation,
            )(x)

            # Create VGG16 model.
            vgg16_model = keras.Model(inputs=base_model.input, outputs=final_x)

            # Freeze layers of base model.
            for layer in base_model.layers:
                layer.trainable = False

        except Exception as e:
            # logger.error(f'Unable to buildVGG16!\n{e}')
            print((f"Unable to buildVGG16!\n{e}"))

        return vgg16_model

    @lD.log(logBase + ".createDataset")
    def createDataset(logger, self):

        """
        Creates the train and validation datasets from a given directory
        of preprocessed images. The train dataset will include data
        augmentation capabilities.

        Parameters
        ----------
        {} : {}


        Returns
        -------
        num_train_imgs : {int}
            Number of images in train dataset.
        num_val_imgs : {int}
            Number of images in validation dataset.
        train_datagen : {tensorflow.python.keras.preprocessing.image.ImageDataGenerator}
            The train data generator.
        train_gen: {tensorflow.python.keras.preprocessing.image.DataFrameIterator}
            The train dataset with labels.
        val_datagen : {tensorflow.python.keras.preprocessing.image.ImageDataGenerator}
            The val data generator.
        val_gen: {tensorflow.python.keras.preprocessing.image.DataFrameIterator}
            The val dataset with labels.
        """

        try:

            # =======================================
            # 1. Get lists of calc and mass filenames
            # =======================================

            mass = []
            calc = []

            for (curdir, dirs, files) in os.walk(
                top=self.createDataset_all_imgs_dir, topdown=False
            ):

                dirs.sort()
                files.sort()

                for f in files:

                    if f.endswith(self.createDataset_extension):

                        if "mass" in f.lower():
                            mass.append(f)
                        elif "calc" in f.lower():
                            calc.append(f)

            # ==========================================
            # 2. Random split paths into train and valid
            # ==========================================

            mass_val_count = round(self.createDataset_val_split * len(mass))
            calc_val_count = round(self.createDataset_val_split * len(calc))

            random.seed(self.RSEED)

            mass_val = random.sample(mass, mass_val_count)
            mass_train = [m for m in mass if m not in mass_val]

            calc_val = random.sample(calc, calc_val_count)
            calc_train = [c for c in calc if c not in calc_val]

            val = mass_val + calc_val
            train = mass_train + calc_train

            random.shuffle(val)
            random.shuffle(train)

            num_train_imgs = len(train)
            num_val_imgs = len(val)

            # ==============================================
            # 3. Create train and test dataframe with labels
            # ==============================================

            # Manual labels --> 1 = Calc, 0 = Mass.

            # Validation dataset.
            val_df = pd.DataFrame(data=val, columns=["filename"])
            val_df["label"] = val_df["filename"].apply(
                lambda x: "calc" if "Calc" in x else "mass"
            )
            val_df["calc"] = val_df["filename"].apply(lambda x: 1 if "Calc" in x else 0)
            val_df["mass"] = val_df["filename"].apply(lambda x: 1 if "Mass" in x else 0)

            # Train dataset.
            train_df = pd.DataFrame(data=train, columns=["filename"])
            train_df["label"] = train_df["filename"].apply(
                lambda x: "calc" if "Calc" in x else "mass"
            )
            train_df["calc"] = train_df["filename"].apply(
                lambda x: 1 if "Calc" in x else 0
            )
            train_df["mass"] = train_df["filename"].apply(
                lambda x: 1 if "Mass" in x else 0
            )

            # ==========================================================
            # 4. Use ImageDataGenerator to create train and val datasets
            # ==========================================================

            brightness_range = (
                self.createDataset_brightness_range[0],
                self.createDataset_brightness_range[1],
            )
            target_size = (self.resnet_input_width, self.resnet_input_height)

            # Define data generators.
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=brightness_range,
            )

            val_datagen = ImageDataGenerator(rescale=1.0 / 255)

            # Get the data.
            #                         !!!
            # Note that we use the "label" column as the labels (i.e.
            # labels = {"calc", "mass"}). Because class_mode="binary", the
            # labels seen by the model are sorted alphanumerically, i.e.
            # {"calc": 0, "mass": 1}.
            #                         !!!

            train_gen = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                directory=self.createDataset_all_imgs_dir,
                x_col="filename",
                y_col="label",
                batch_size=self.createDataset_batch_size,
                color_mode="rgb",
                class_mode="binary",
                target_size=target_size,
                shuffle=True,
                seed=self.RSEED,
            )

            val_gen = val_datagen.flow_from_dataframe(
                dataframe=val_df,
                directory=self.createDataset_all_imgs_dir,
                x_col="filename",
                y_col="label",
                batch_size=self.createDataset_batch_size,
                color_mode="rgb",
                class_mode="binary",
                target_size=target_size,
                shuffle=True,
                seed=self.RSEED,
            )

        except Exception as e:
            # logger.error(f'Unable to createDataset!\n{e}')
            print((f"Unable to createDataset!\n{e}"))

        return (train_df, num_train_imgs, train_datagen, train_gen), (
            val_df,
            num_val_imgs,
            val_datagen,
            val_gen,
        )

    @lD.log(logBase + ".compile_")
    def compile_(logger, self, model):

        """
        Compiles the resnet50_model created using self.buildResNet50().

        Parameters
        ----------
        model : {tf.Function}
            The ResNet50 model that self.buildResNet50() returns.

        Returns
        -------
        resnet50_model: {}
            The compiled ResNet50 model.
        """

        try:
            if "adam" in self.compile_optimiser.lower():
                optimizer = keras.optimizers.Adam(
                    learning_rate=self.compile_learning_rate
                )
            elif "sgd" in self.compile_optimiser.lower():
                optimizer = keras.optimizers.SGD(
                    learning_rate=self.compile_learning_rate
                )
            metrics = ["accuracy", keras.metrics.Recall(), keras.metrics.Precision()]
            loss = self.compile_loss

            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        except Exception as e:
            # logger.error(f'Unable to compile!\n{e}')
            print((f"Unable to compile!\n{e}"))

        return model

    @lD.log(logBase + ".createCallbacks")
    def createCallbacks(
        logger, self, model_time, ckpt_folder, tensorboard_folder, csv_logger_folder
    ):

        """
        asdf.

        Parameters
        ----------
        {model_time : {str}
            Time where self.train() was called.
        {ckpt_folder} : {path}
            Folder where checkpoints are to be saved in.
        {tensorboard_folder} : {path}
            Folder where tensorlogs are to be saved in.
        {csv_logger_folder} : {path}
            Folder where csv logs are to be saved in.

        Returns
        -------
        callbacks: {}
            The callbacks to be used for training.
        """

        try:

            # Checkpoint
            ckpt_path = (
                ckpt_folder
                + f"/{model_time}"
                # + "_Epoch-{epoch:03d}"
                # + "_Acc-{accuracy:.8f}"
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

        except Exception as e:
            # logger.error(f'Unable to createCallbacks!\n{e}')
            print((f"Unable to createCallbacks!\n{e}"))

        return callbacks

    @lD.log(logBase + ".train")
    def train(logger, self):

        """
        Creates the ResNet50 model using self.buildResNet50(), creates the train
        and validation dataset, compiles the model, then trains the model.

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

            # Parent folder
            model_folder = os.path.join(self.results_dir, model_time)
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

            # Saved model folder
            saved_model_folder = os.path.join(model_folder, "saved_model")
            os.makedirs(saved_model_folder)

            # Json params folder
            model_params_folder = os.path.join(model_folder, "model_params")
            os.makedirs(model_params_folder)

            # ===============
            #  Create dataset
            # ===============
            (train_df, num_train_imgs, train_datagen, train_gen), (
                val_df,
                num_val_imgs,
                val_datagen,
                val_gen,
            ) = self.createDataset()

            # Save train_df and val_df,
            # (so that we know which images are
            # in the train and validation sets)
            train_df.to_csv(
                os.path.join(model_params_folder, "train_set.csv"), index=False
            )
            val_df.to_csv(os.path.join(model_params_folder, "val_set.csv"), index=False)

            # =============
            #  Build model
            # =============
            if self.which_model.lower() == "resnet50":
                model = self.buildResNet50()
            elif self.which_model.lower() == "vgg16":
                model = self.buildVGG16()

            # ==========
            #  Compile
            # ==========
            model = self.compile_(model=model)
            print(model.summary())

            # ===========
            #  Callbacks
            # ===========
            callbacks = self.createCallbacks(
                model_time=model_time,
                ckpt_folder=ckpt_folder,
                csv_logger_folder=csv_logger_folder,
                tensorboard_folder=tensorboard_folder,
            )

            # =====
            #  Fit
            # =====

            # Calculate train and validation steps per epoch.
            train_steps = num_train_imgs // self.createDataset_batch_size
            val_steps = num_val_imgs // self.createDataset_batch_size

            if num_train_imgs % self.createDataset_batch_size != 0:
                train_steps += 1
            if num_val_imgs % self.createDataset_batch_size != 0:
                val_steps += 1

            print()
            print(f"Model time = {model_time}")
            print(f"Size of training set = {num_train_imgs}")
            print(f"Size of test set = {num_val_imgs}")
            print(f"Number of epochs = {self.num_epochs}")
            print(f"Batch size = {self.createDataset_batch_size}")
            print(f"Number of training steps per epoch = {train_steps}")
            print(f"Number of test steps per epoch = {val_steps}")
            print()

            # Fit!
            model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                epochs=self.num_epochs,
                validation_data=val_gen,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=1,
            )

            # ==========================
            #  Save relevant model infos
            # ==========================

            # Save model
            saved_model_path = os.path.join(saved_model_folder, model_time)
            model.save(saved_model_path)

            # Save model params
            model_params_path = os.path.join(model_params_folder, "model_params.json")
            with open(model_params_path, "w") as f:
                json.dump(config_resnet, f)

        except Exception as e:
            # logger.error(f'Unable to train!\n{e}')
            print((f"Unable to train!\n{e}"))

        return