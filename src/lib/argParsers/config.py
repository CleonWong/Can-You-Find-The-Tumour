from logs import logDecorator as lD
import jsonref

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.argParsers.config'

@lD.log(logBase + '.parsersAdd')
def addParsers(logger, parser):
    '''add argument parsers specific to the ``config/config.json`` file
    
    This function is kgoing to add argument parsers specific to the 
    ``config/config.json`` file. This file has several options for 
    logging data. This information will be 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    parser : {argparse.ArgumentParser instance}
        An instance of ``argparse.ArgumentParser()`` that will be
        used for parsing the command line arguments specific to the 
        config file
    
    Returns
    -------
    argparse.ArgumentParser instance
        The same parser argument to which new CLI arguments have been
        appended
    '''
    
    parser.add_argument("--logging_level", 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="change the logging level")
    # parser.add_argument("--logging_specs_file_todo", 
    #     action="store_true",
    #     help="allow file logging")
    parser.add_argument("--logging_specs_file_logFolder", 
        type = str,
        help = "folder in which to log files")
    parser.add_argument("--logging_specs_stdout_todo", 
        action="store_true",
        help="allow stdout logging")
    parser.add_argument("--logging_specs_logstash_todo", 
        action="store_true",
        help="allow logstash logging")
    parser.add_argument("--logging_specs_logstash_version", 
        type = int,
        help = "version for the logstash server")
    parser.add_argument("--logging_specs_logstash_port", 
        type = int,
        help = "port for the logstash server")
    parser.add_argument("--logging_specs_logstash_host", 
        type = str,
        help = "hostname for the logstash server")

    return parser

@lD.log(logBase + '.decodeParser')
def decodeParser(logger, args):
    '''generate a dictionary from the parsed args
    
    The parsed args may/may not be present. When they are
    present, they are pretty hard to use. For this reason,
    this function is going to convert the result into
    something meaningful.
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    args : {args Namespace}
        parsed arguments from the command line
    
    Returns
    -------
    dict
        Dictionary that converts the arguments into something
        meaningful
    '''


    values = {
        'specs': {
            'file': {},
            'stdout': {},
            'logstash' : {}
        }
    }
    
    try:
        if args.logging_level is not None:
            values['level'] = args.logging_level
    except Exception as e:
        logger.error('Unable to decode the argument logging_level :{}'.format(
            e))
    # try:
    #     if args.logging_specs_file_todo is not None:
    #         values['specs']['file']['todo'] = args.logging_specs_file_todo
    # except Exception as e:
    #     logger.error('Unable to decode the argument logging_specs_file_todo :{}'.format(
    #         e))
    try:
        if args.logging_specs_file_logFolder is not None:
            values['specs']['file']['logFolder'] = args.logging_specs_file_logFolder
    except Exception as e:
        logger.error('Unable to decode the argument logging_specs_file_logFolder :{}'.format(
            e))
    try:
        if args.logging_specs_stdout_todo is not None:
            values['specs']['stdout']['todo'] = args.logging_specs_stdout_todo
    except Exception as e:
        logger.error('Unable to decode the argument logging_specs_stdout_todo :{}'.format(
            e))
    try:
        if args.logging_specs_logstash_todo is not None:
            values['specs']['logstash']['todo'] = args.logging_specs_logstash_todo
    except Exception as e:
        logger.error('Unable to decode the argument logging_specs_logstash_todo :{}'.format(
            e))
    try:
        if args.logging_specs_logstash_version is not None:
            values['specs']['logstash']['version'] = args.logging_specs_logstash_version
    except Exception as e:
        logger.error('Unable to decode the argument logging_specs_logstash_version :{}'.format(
            e))
    try:
        if args.logging_specs_logstash_port is not None:
            values['specs']['logstash']['port'] = args.logging_specs_logstash_port
    except Exception as e:
        logger.error('Unable to decode the argument logging_specs_logstash_port :{}'.format(
            e))
    try:
        if args.logging_specs_logstash_host is not None:
            values['specs']['logstash']['host'] = args.logging_specs_logstash_host
    except Exception as e:
        logger.error('Unable to decode the argument logging_specs_logstash_host :{}'.format(
            e))
    
    return values

