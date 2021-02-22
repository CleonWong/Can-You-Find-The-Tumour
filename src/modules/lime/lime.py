import os
import pprint
import jsonref
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.lime.lime"
config_lime = jsonref.load(open("../config/modules/lime.json"))


@lD.log(logBase + ".loadModel")
def loadModel(logger, model_dir):

    """
    This function loads the already-trained classification model that we want
    to use for executing Grad-CAM.

    Parameters
    ----------
    model_dir : {path}
        The path of the folder where the already-trained TensorFlow SavedModel
        is saved.

    Returns
    -------
    model : {Keras model instance}
        The loaded already-trained TensorFlow SavedModel.
    """

    try:
        model = tf.keras.models.load_model(filepath=model_dir)

    except Exception as e:
        # logger.error(f'Unable to loadModel!\n{e}')
        print((f"Unable to loadModel!\n{e}"))

    return model


@lD.log(logBase + ".gradModel")
def lime(logger, img, model, num_samples, num_features, random_seed):

    """
    bla bla bla

    Parameters
    ----------
    img : {numpy.ndarray}
        A single image that is a numpy array in the shape of ()
    model : {Keras model instance}
        The loaded already-trained TensorFlow SavedModel, as returned by the
        function `loadModel()`.
    num_samples : {int}
        The size of the neighborhood to learn the linear model
    num_features : {int}
        The number of superpixels to include in explanation.

    Returns
    -------
    temp_img : {numpy.ndarray}
        3D numpy array of `img`'s explanation.
    mask : {numpy.ndarray}
        2D numpy array that can be used with skimage.segmentation.mark_boundaries.
    """

    try:

        # Instantiate explainer.
        explainer = lime_image.LimeImageExplainer()

        # Perform lime.
        explanation = explainer.explain_instance(
            img,
            classifier_fn=model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            random_seed=random_seed,
        )
        temp_img, mask = explanation.get_image_and_mask(
            label=0, positive_only=True, hide_rest=True, num_features=num_features
        )

    except Exception as e:
        # logger.error(f'Unable to lime!\n{e}')
        print((f"Unable to lime!\n{e}"))

    return temp_img, mask


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for lime module.

    bla bla bla

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dictionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    """

    print("=" * 30)
    print("Main function of lime module.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Load parameters.
    model_dir = Path(config_lime["model_dir"])
    img_folder = Path(config_lime["img_folder"])
    img_width = config_lime["img_width"]
    img_height = config_lime["img_height"]
    seed = config_lime["seed"]
    num_samples = config_lime["num_samples"]
    num_features = config_lime["num_features"]
    to_lime_csv = Path(config_lime["to_lime_csv"])
    saveto_folder = Path(config_lime["saveto_folder"])

    # ============
    #  Load model
    # ============
    model = loadModel(model_dir=model_dir)

    # =====================
    #  Loop and apply LIME
    # =====================

    # Load .csv that contains filenames of images to do Grad-CAM on.
    to_lime_df = pd.read_csv(to_lime_csv, header=0)

    for row in to_lime_df.itertuples():
        # ------------
        #  Load image
        # ------------
        img_filename = row.filename
        img_path = os.path.join(img_folder, img_filename)

        # Set dsize = (img width, img height).
        dsize = (img_width, img_height)

        # Load image.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src=img, dsize=dsize)

        # Min max normalise to [0, 1].
        img = (img - img.min()) / (img.max() - img.min() + 1.0e-10)

        # Stack grayscale image to make channels=3.
        img = np.stack([img, img, img], axis=-1)

        # Make img shape into (1, img_width, img_height, 3).
        img = img[np.newaxis, :]

        # ---------
        #  Do LIME
        # ---------
        temp_img_list = []
        mask_list = []

        for n in num_features:
            temp_img, mask = lime(
                img=img[0],
                model=model,
                num_samples=num_samples,
                num_features=n,
                random_seed=seed,
            )
            temp_img_list.append(temp_img)
            mask_list.append(mask)

        # --------------------
        #  Plot and save LIME
        # --------------------
        n = len(temp_img_list)

        fig, ax = plt.subplots(nrows=1, ncols=n + 1, figsize=(n * 5, 8))

        # Plot original image.
        ax[0].imshow(img[0], cmap="gray")
        ax[0].set_title(f"{img_filename}\nProba: {row.predicted_proba}")
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)

        # Plot LIME outputs.
        for i, ax in enumerate(fig.axes):
            if i == 0:
                continue  # To skip plotting on ax[0].
            else:
                ax.imshow(mark_boundaries(temp_img_list[i - 1], mask_list[i - 1]))
                ax.set_title(f"LIME with top {num_features[i - 1]} superpixels")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.tight_layout()

        lime_filename = img_filename.replace("___PRE.png", "___LIME.png")
        fname = os.path.join(saveto_folder, lime_filename)
        plt.savefig(fname=fname, dpi=300)

    print()
    print("Getting out of lime module.")
    print("-" * 30)

    return
