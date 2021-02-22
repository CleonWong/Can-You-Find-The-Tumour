from logs import logDecorator as lD
from lib.unet import unetVgg16
import jsonref, os
import numpy as np
import tensorflow as tf

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.trainUnet.trainUnet"

config_unet = jsonref.load(open("../config/modules/unet.json"))
# data_dir = configVgg["dataPath"]
# tfPath = os.path.join(dataPath, "tfRecords")
# files = np.array([os.path.join(tfPath, x) for x in os.listdir(tfPath)])


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for trainUnet.

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
    print("Main function of trainUnet.")
    print("=" * 30)

    # Seeding.
    seed = config_unet["seed"]
    tf.random.set_seed(seed)

    # Instantiate custom unet model class.
    unet = unetVgg16.unetVgg16()

    # Train the model.
    unet.train()

    print()
    print("Getting out of trainUnet.")
    print("-" * 30)
    print()

    return