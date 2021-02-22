from lib.argParsers import config as cf

from logs import logDecorator as lD
import jsonref, copy

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.argParsers.addAllParsers'

@lD.log(logBase + '.parsersAdd')
def parsersAdd(logger, parser):
    '''add all available CLI arguments to the parser
    
    This function is going to add all available parser
    information into the provided parser argument.
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    parser : {argparse.ArgumentParser instance}
        An instance of ``argparse.ArgumentParser()`` that will be
        used for parsing the command line arguments.
    
    Returns
    -------
    ``argparse.ArgumentParser()`` instance
        This is a ``argparse.ArgumentParser()`` instance that captures
        all the optional argument options that have been passed to 
        the instance
    '''

    parser = cf.addParsers(parser)

    return parser

@lD.log(logBase + '.updateArgs')
def updateArgs(logger, defaultDict, claDict):
    '''helper function for decoding arguments
    
    This function takes the dictionary provided by the
    namespace arguments, and updates the dictionary that
    needs parsing, in a meaningful manner. This allows
    ``str``, ``bool``, ``int``, ``float``, ``complex`` 
    and ``dict`` arguments to be changed. Make sure that
    you use it with caution. If you are unsure what this
    is going to return, just role your own parser.
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    defaultDict : {[type]}
        [description]
    claDict : {[type]}
        [description]
    
    Returns
    -------
    [type]
        [description]
    '''

    for d in defaultDict:
        if d not in claDict:
            continue

        t = type(defaultDict[d])
        if t is bool:
            defaultDict[d] = (claDict[d] or defaultDict[d])
            continue

        if  any([t is m for m in [str, int, float, complex]]):
            defaultDict[d] = claDict[d]
            continue

        if t is dict:
            defaultDict[d] = updateArgs(defaultDict[d], claDict[d])
            continue

        logger.error('Unable to process type: [{}] for [{}]'.format(t, d))

    return defaultDict

@lD.log(logBase + '.decodeParsers')
def decodeParsers(logger, args):
    '''convert the parser namespace into a dict
    
    This takes the parsed arguments and converts the values
    into a dictionary that can be used ...
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    args : {a parsed object}
        A Namespace that contains the values of the parsed
        arguments according to the values provided.

    Returns
    -------
    dict
        A doctionary containing a list of all the parsers
        converted into their respective sub dictionaries
    '''

    allConfigs = {}

    configCLA = {}
    configCLA['logging'] = cf.decodeParser(args)

    allConfigs['config'] = configCLA

    return allConfigs

