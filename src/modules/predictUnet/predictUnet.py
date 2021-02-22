from logs import logDecorator as lD
from lib.unet import unetVgg16
import jsonref, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.predictUnet.predictUnet"

config_predictUnet = jsonref.load(open("../config/modules/predictUnet.json"))


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for predictUnet.

    This function finishes all the tasks for the
    main function. This is a way in which a
    particular module is going to be executed.

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
    print("Main function of predictUnet.")
    print("=" * 30)

    # Get parameters from .json files.
    ckpt_path = config_predictUnet["ckpt_path"]
    to_predict_x = Path(config_predictUnet["to_predict_x"])
    to_predict_y = Path(config_predictUnet["to_predict_y"])
    extension = config_predictUnet["extension"]
    target_size = (
        config_predictUnet["target_size"],
        config_predictUnet["target_size"],
    )
    save_predicted = Path(config_predictUnet["save_predicted"])

    # Seeding.
    seed = config_predictUnet["seed"]
    tf.random.set_seed(seed)

    # ====================
    #  Create test images
    # ====================
    # Get paths to individual images.
    test_x, test_y = unetVgg16.unetVgg16().datasetPaths(
        full_img_dir=to_predict_x,
        mask_img_dir=to_predict_y,
        extension=extension,
    )

    # Read FULL images.
    test_imgs = [
        unetVgg16.unetVgg16().loadFullImg(path=path, dsize=target_size)
        for path in test_x
    ]
    test_imgs = np.array(test_imgs, dtype=np.float64)

    # Read MASK images.
    test_masks = [
        unetVgg16.unetVgg16().loadMaskImg(path=path, dsize=target_size)
        for path in test_y
    ]
    test_masks = np.array(test_masks, dtype=np.float64)

    # ============
    #  Load model
    # ============
    # Method 1:
    # ---------
    # model = keras.models.load_model(
    #     filepath=ckpt_path,
    #     compile=True,
    #     custom_objects={"iouMetric": unetVgg16.unetVgg16().iouMetric},
    # )
    # print(model.summary())

    # Method 2:
    # ---------
    model = unetVgg16.unetVgg16()
    unet = model.buildUnet(dropout_training=False)
    unet = model.compile_(model=unet)

    # Load pre-trained weights from desired checkpoint file.
    latest = tf.train.latest_checkpoint(ckpt_path)
    print(latest)
    unet.load_weights(filepath=latest)

    # =========
    #  Predict
    # =========
    predicted_outputs = unet.predict(x=test_imgs, batch_size=len(test_imgs))

    for i in range(len(predicted_outputs)):

        # =======================================
        #  Save predicted numpy arrays as images
        # =======================================

        # Get patient ID
        filename = os.path.basename(test_x[i])
        patientID = filename.replace("___PRE" + extension, "")

        # Get save path
        filename_new_1 = patientID + "___PREDICTED" + extension
        filename_new_3 = patientID + "___PREDICTED_ALL" + extension
        save_path_1 = os.path.join(save_predicted, "predmask_only", filename_new_1)
        save_path_3 = os.path.join(
            save_predicted, "full_truemask_predmask", filename_new_3
        )

        # Plot image and save
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
        ax[0].imshow(test_imgs[i], cmap="gray")
        ax[1].imshow(test_masks[i], cmap="gray")
        ax[2].imshow(predicted_outputs[i] * 255, cmap="gray")

        print(predicted_outputs[i].min(), predicted_outputs[i].max())

        # Set title and remove axes
        patientID_noFULL = patientID.replace("_FULL", "")
        ax[0].set_title(f"{patientID_noFULL} - Full scan")
        ax[1].set_title(f"{patientID_noFULL} - Ground truth mask")
        ax[2].set_title(f"{patientID_noFULL} - Predicted mask")
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        plt.tight_layout()

        # Save mammogram, true mask and predicted mask together.
        # ------------------------------------------------------
        plt.savefig(fname=save_path_3, dpi=300)

        # Save just predicted mask.
        # -------------------------
        cv2.imwrite(filename=save_path_1, img=predicted_outputs[i] * 255)

    print()
    print("Getting out of predictUnet.")
    print("-" * 30)
    print()

    return