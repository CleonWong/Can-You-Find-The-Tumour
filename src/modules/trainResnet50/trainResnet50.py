from logs import logDecorator as lD
from lib.resnet import resnet50
import jsonref, os
import numpy as np
import tensorflow as tf

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.trainResnet50.trainResnet50"

config_resnet = jsonref.load(open("../config/modules/resnet50.json"))


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for trainResnet50.

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
    print("Main function of trainResnet50.")
    print("=" * 30)

    # Seeding.
    seed = config_resnet["seed"]
    tf.random.set_seed(seed)

    # Instantiate custom unet model class.
    resnet50_ = resnet50.resnet50()

    # Train the model.
    resnet50_.train()

    print()
    print("Getting out of trainResnet50.")
    print("-" * 30)
    print()

    return