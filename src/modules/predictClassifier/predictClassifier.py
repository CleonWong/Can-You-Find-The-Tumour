from logs import logDecorator as lD
import jsonref, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import cv2

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.predictClassifier.predictClassifier"

config_predictClassifier = jsonref.load(
    open("../config/modules/predictClassifier.json")
)


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for predictClassifier.

    This function finishes all the tasks for the
    main function. This is a way in which a
    particular module is going to be executed.

    predictClassifier makes classification predictions
    on images using a specified trained model. The
    classification is binary, where  {Calc=1, Mass=0}.
    The single output neuron classifies for the class
    "Calc". The trained model is loaded from a TensorFlow
    SavedModel instance.

    The train and validation set split can be found in
    the `train_set.csv` and `val_set.csv` files located
    in the relevant result folder.

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    """

    print("=" * 30)
    print("Main function of predictClassifier.")
    print("=" * 30)

    # Get parameters from .json files.
    to_predict_csv_path = Path(config_predictClassifier["to_predict_csv_path"])
    savedmodel_path = Path(config_predictClassifier["savedmodel_path"])
    img_folder = Path(config_predictClassifier["img_folder"])
    save_predicted_csv = Path(config_predictClassifier["save_predicted_csv"])
    target_size = config_predictClassifier["target_size"]

    # Seeding.
    seed = config_predictClassifier["seed"]
    tf.random.set_seed(seed)

    # ================================================
    #  Load .csv that contains image paths and labels
    # ================================================

    to_predict_df = pd.read_csv(to_predict_csv_path, header=0)

    # ============
    #  Load model
    # ============

    # ----------
    #  Method 1
    # ----------
    model = keras.models.load_model(filepath=savedmodel_path)
    print(model.summary())

    # ----------
    #  Method 2
    # ----------
    # model = unetVgg16.unetVgg16()
    # unet = model.buildUnet(dropout_training=False)
    # unet = model.compile_(model=unet)
    # Load pre-trained weights from desired checkpoint file.
    # latest = tf.train.latest_checkpoint(ckpt_path)
    # print(latest)
    # unet.load_weights(filepath=latest)

    # =========
    #  Predict
    # =========

    dsize = (target_size, target_size)
    to_predict_df["predicted_proba"] = np.nan

    # Loop through each filepath in `to_predict_df` and predict individually.
    for row in to_predict_df.itertuples():

        # ---------------------------
        #  Step 1: Get path to image
        # ---------------------------
        img_filename = row.filename
        img_path = os.path.join(img_folder, img_filename)

        # -----------------------------------
        #  Step 2: Load image as numpy array
        # -----------------------------------
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src=img, dsize=dsize)

        # Min max normalise to [0, 1].
        img = (img - img.min()) / (img.max() - img.min())

        # Stack grayscale image to make channels=3.
        img = np.stack([img, img, img], axis=-1)

        # Make img shape into (1, 224, 224, 3).
        img = img[np.newaxis, :]

        # -----------------
        #  Step 3: Predict
        # -----------------
        proba = model.predict(x=img)
        proba = proba[0][0]

        # -------------------------------------------------------------------
        #  Step 4: Append predicted probability (`proba`) to `to_predict_df`
        # -------------------------------------------------------------------
        to_predict_df.loc[row.Index, "predicted_proba"] = proba

    # ======================
    #  Save `to_predict_df`
    # ======================

    save_path = os.path.join(save_predicted_csv, "predicted_proba.csv")
    to_predict_df.to_csv(save_path, index=False)

    print()
    print("Getting out of predictClassifier.")
    print("-" * 30)
    print()

    return