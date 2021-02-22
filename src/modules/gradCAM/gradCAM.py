import os
import pprint
import jsonref
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.gradCAM.gradCAM"
config_gradCAM = jsonref.load(open("../config/modules/gradCAM.json"))


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
def gradModel(logger, model, layer_name):

    """
    This function creates a new model called `grad_model` from `model`.
    `grad_model` is almost the same as the inputted `model`. The only difference
    is that on top of also outputting the predicted probabilities that we would
    get from the standard `model`, `grad_model` also outputs the feature maps
    from the targeted conv layer specified by `layer_name`.

    Parameters
    ----------
    model : {Keras model instance}
        The loaded already-trained TensorFlow SavedModel, as returned by the
        function `loadModel()`.

    Returns
    -------
    grad_model : {Keras model instance}
        The model described in the doc string.
    """

    try:

        # Get the target convolution layer.
        conv_layer = model.get_layer(layer_name)

        # Then create a model that goes up to only that `conv_layer` layer.
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.outputs],
            name="grad_model",
        )

    except Exception as e:
        # logger.error(f'Unable to gradModel!\n{e}')
        print((f"Unable to gradModel!\n{e}"))

    return grad_model


@lD.log(logBase + ".gradCAM")
def gradCAM(logger, grad_model, img, thresh, resize):

    """
    This function generates the Grad-CAM heatmap for the given `img`.

    Parameters
    ----------
    grad_model : {Keras model instance}
        The model returned by the function `gradModel()`.
    img : {numpy.array}
        The desired image to do Grad-CAM on.
    thresh : {float}
        The threshold for clipping the gradcam heatmap. Ranges between [0, 1].
    resize : {tuple}
        The size to resample the gradcam to (img_width, img_height).

    Returns
    -------
    gradcam : {numpy.array}
        The Grad-CAM heatmap for `img`.
    """

    try:

        # ====================
        #  Calculate gradient
        # ====================

        # Get the score (y_c) for the target class.
        with tf.GradientTape() as tape:
            # `conv_outputs` is the outputed feature maps from the final conv layer.
            # `predictions` is the computed loss value
            last_conv_outputs, predictions = grad_model(img)
            y_c = predictions[0][0]

        # =============
        #  Do Grad-CAM
        # =============

        # --------
        #  Step 1
        # --------
        # Calculate the partial derivative of the model outputs (y_c) wrt the feature map activations of `conv_layer`.
        # Each one of these gradients represents the connection from one of the pixels in the feature maps to the output neuron representing the target class.
        gradients = tape.gradient(y_c, last_conv_outputs)[0]

        # --------
        #  Step 2
        # --------
        # Do global average pooling on the gradients to get the alphas.
        pooled_gradients = tf.reduce_mean(gradients, axis=[0, 1])

        # --------
        #  Step 3
        # --------
        # Do element-wise multiplication between each feature map and its corresponding alpha.
        last_conv_outputs = last_conv_outputs.numpy()[0]
        pooled_gradients = pooled_gradients.numpy()

        # Element-wise multiplication.
        for i in range(pooled_gradients.shape[-1]):
            last_conv_outputs[:, :, i] *= pooled_gradients[i]

        # Sum all feature maps to get a single 2D array.
        gradcam = np.sum(last_conv_outputs, axis=-1)

        # Apply RELU (i.e. max(0, element)).
        gradcam = np.clip(a=gradcam, a_min=0, a_max=gradcam.max())

        # ----------------------
        #  Clean up the heatmap
        # ----------------------
        # Min-max normalise.
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1.0e-10)

        # Threshold clipping.
        gradcam[gradcam < thresh] = 0
        # gradcam = np.clip(a=gradcam, a_min=thresh, a_max=gradcam.max())

        # Resize to (224, 224).
        gradcam = cv2.resize(src=gradcam, dsize=resize)

    except Exception as e:
        # logger.error(f'Unable to gradCAM!\n{e}')
        print((f"Unable to gradCAM!\n{e}"))

    return gradcam


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for gradCAM module.

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
    print("Main function of gradCAM module.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Load parameters.
    model_dir = Path(config_gradCAM["model_dir"])
    layer_name = config_gradCAM["layer_name"]
    img_width = config_gradCAM["img_width"]
    img_height = config_gradCAM["img_height"]
    thresh = config_gradCAM["thresh"]
    img_folder = Path(config_gradCAM["img_folder"])
    to_gradcam_csv = Path(config_gradCAM["to_gradcam_csv"])
    saveto_folder = Path(config_gradCAM["saveto_folder"])

    # ============
    #  Load model
    # ============
    model = loadModel(model_dir=model_dir)

    # =====================
    #  Set up for Grad-CAM
    # =====================

    # Get `grad_model`
    grad_model = gradModel(model=model, layer_name=layer_name)
    print(grad_model.summary())

    # =========================
    #  Loop and apply Grad-CAM
    # =========================

    # Load .csv that contains filenames of images to do Grad-CAM on.
    to_gradcam_df = pd.read_csv(to_gradcam_csv, header=0)

    dsize = (img_width, img_height)

    for row in to_gradcam_df.itertuples():
        # ------------
        #  Load image
        # ------------
        img_filename = row.filename
        img_path = os.path.join(img_folder, img_filename)

        # Load image.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src=img, dsize=dsize)

        # Min max normalise to [0, 1].
        img = (img - img.min()) / (img.max() - img.min() + 1.0e-10)

        # Stack grayscale image to make channels=3.
        img = np.stack([img, img, img], axis=-1)

        # Make img shape into (1, img_width, img_height, 3).
        img = img[np.newaxis, :]

        # -------------
        #  Do Grad-CAM
        # -------------
        gradcam_list = [
            gradCAM(grad_model=grad_model, img=img, thresh=t, resize=dsize)
            for t in thresh
        ]
        # gradcam = gradCAM(grad_model=grad_model, img=img, thresh=thresh, resize=dsize)

        # ------------------------
        #  Plot and save Grad-CAM
        # ------------------------
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 13))

        ax[0][0].imshow(img[0], cmap="gray")
        # ax[1].imshow(gradcam)
        # ax[2].imshow(img[0], cmap="gray")
        # ax[2].imshow(gradcam, alpha=0.5)
        ax[0][1].imshow(gradcam_list[0])
        ax[0][2].imshow(gradcam_list[1])
        ax[0][3].imshow(gradcam_list[2])
        ax[0][4].imshow(gradcam_list[3])

        ax[1][0].imshow(np.ones_like(img[0]), cmap="gray")
        ax[1][1].imshow(img[0], cmap="gray")
        ax[1][1].imshow(gradcam_list[0], alpha=0.5)
        ax[1][2].imshow(img[0], cmap="gray")
        ax[1][2].imshow(gradcam_list[1], alpha=0.5)
        ax[1][3].imshow(img[0], cmap="gray")
        ax[1][3].imshow(gradcam_list[2], alpha=0.5)
        ax[1][4].imshow(img[0], cmap="gray")
        ax[1][4].imshow(gradcam_list[3], alpha=0.5)

        ax[0][0].set_title(f"{img_filename}\nProba: {row.predicted_proba}")
        ax[0][1].set_title(f"Grad-CAM heatmap\nThreshold = {thresh[0]}")
        ax[0][2].set_title(f"Grad-CAM heatmap\nThreshold = {thresh[1]}")
        ax[0][3].set_title(f"Grad-CAM heatmap\nThreshold = {thresh[2]}")
        ax[0][4].set_title(f"Grad-CAM heatmap\nThreshold = {thresh[3]}")
        # ax[2].set_title(f"Grad-CAM heatmap overlay")

        for i, ax in enumerate(fig.axes):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()

        gradcam_filename = img_filename.replace("___PRE.png", "___GRADCAM.png")
        fname = os.path.join(saveto_folder, gradcam_filename)
        plt.savefig(fname=fname, dpi=300)

    print()
    print("Getting out of gradCAM module.")
    print("-" * 30)

    return
