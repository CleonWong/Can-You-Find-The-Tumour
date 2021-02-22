import jsonref, argparse

from importlib import util
from logs import logDecorator as lD
from lib.testLib import simpleLib as sL
from lib.argParsers import addAllParsers as aP

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"]
logLevel = config["logging"]["level"]
logSpecs = config["logging"]["specs"]


@lD.log(logBase + ".importModules")
def importModules(logger, resultsDict):
    """import and execute required modules

    This function is used for importing all the
    modules as defined in the ../config/modules.json
    file and executing the main function within it
    if present. In error, it fails gracefully ...

    Parameters
    ----------
    logger : {logging.Logger}
        logger module for logging information
    """
    modules = jsonref.load(open("../config/modules.json"))

    # update modules in the right order. Also get rid of the frivilous
    # modules
    if resultsDict["modules"] is not None:
        tempModules = []
        for m in resultsDict["modules"]:
            toAdd = [n for n in modules if n["moduleName"] == m][0]
            tempModules.append(toAdd)

        modules = tempModules

    for m in modules:

        if resultsDict["modules"] is None:

            # skip based upon modules.json
            logger.info("Obtaining module information from modules.json")
            try:
                if not m["execute"]:
                    logger.info("Module {} is being skipped".format(m["moduleName"]))
                    continue
            except Exception as e:
                logger.error(
                    f"Unable to check whether module the module should be skipped: {e}"
                )
                logger.error(f"this module is being skipped")
                continue

        else:

            # skip based upon CLI
            try:
                if m["moduleName"] not in resultsDict["modules"]:
                    logger.info(
                        f"{m} not present within the list of CLI modules. Module is skipped"
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Unable to determine whether this module should be skipped: {e}.\n Module is being skipped."
                )
                continue

        try:
            name, path = m["moduleName"], m["path"]
            logger.info("Module {} is being executed".format(name))

            module_spec = util.spec_from_file_location(name, path)
            module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            module.main(resultsDict)
        except Exception as e:
            print("Unable to load module: {}->{}\n{}".format(name, path, str(e)))

    return


def main(logger, resultsDict):
    """main program

    This is the place where the entire program is going
    to be generated.
    """

    # First import all the modules, and run
    # them
    # ------------------------------------
    importModules(resultsDict)

    # Lets just create a simple testing
    # for other functions to follow
    # -----------------------------------

    return


if __name__ == "__main__":

    # Let us add an argument parser here
    parser = argparse.ArgumentParser(description="mammogram-cv command line arguments")

    # Add the modules here
    modules = jsonref.load(open("../config/modules.json"))
    modules = [m["moduleName"] for m in modules]
    parser.add_argument(
        "-m",
        "--module",
        action="append",
        type=str,
        choices=modules,
        help="""Add modules to run over here. Multiple modules can be run
        simply by adding multiple strings over here. Make sure that the 
        available choices are reflected in the choices section""",
    )

    parser = aP.parsersAdd(parser)
    results = parser.parse_args()
    resultsDict = aP.decodeParsers(results)

    if results.module is not None:
        resultsDict["modules"] = results.module
    else:
        resultsDict["modules"] = None

    # ---------------------------------------------------
    # We need to explicitely define the logging here
    # rather than as a decorator, bacause we have
    # fundamentally changed the way in which logging
    # is done here
    # ---------------------------------------------------
    logSpecs = aP.updateArgs(logSpecs, resultsDict["config"]["logging"]["specs"])
    try:
        logLevel = resultsDict["config"]["logging"]["level"]
    except Exception as e:
        print("Logging level taking from the config file: {}".format(logLevel))

    logInit = lD.logInit(logBase, logLevel, logSpecs)
    main = logInit(main)

    main(resultsDict)