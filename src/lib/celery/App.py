from logs import logDecorator as lD
import jsonref, psycopg2
from celery import Celery
import logging

from datetime import datetime as dt

from celery.signals import after_setup_logger


config   = jsonref.load(open('../config/config.json'))
logBase  = config['logging']['logBase']
logLevel = config['logging']['level']
logSpecs = config['logging']['specs']
cConfig  = jsonref.load(open('../config/celery.json'))

logger = logging.getLogger(logBase)

app = Celery( 
    cConfig['base']['name'], 
    broker  = cConfig['base']['BROKER_URL'],
    backend = cConfig['base']['BACKEND_URL'],
    include = cConfig['base']['include'])

app.conf.update( **cConfig['extra'] )

# ------------------------------------------------------------------
# https://www.distributedpython.com/2018/08/28/celery-logging/
# help about loggign is obtained from here ...
# ------------------------------------------------------------------
@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    now       = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Generate a file handler if necessary
    if ('file' in logSpecs) and logSpecs['file']['todo']:
        fH  = logging.FileHandler( '{}/celery_{}.log'.format(
            logSpecs['file']['logFolder'], now) )
        
        fH.setFormatter(formatter)
        logger.addHandler(fH)

    # Generate a file handler if necessary
    if ('stdout' in logSpecs) and logSpecs['stdout']['todo']:
        cH = logging.StreamHandler(sys.stdout)
        cH.setFormatter(formatter)
        logger.addHandler(cH)

    # Generate a file handler if necessary
    if ('logstash' in logSpecs) and logSpecs['logstash']['todo']:

        tags = [ 'celeryTest' , now]

        if 'tags' in logSpecs['logstash']:
            tags += logSpecs['logstash']['tags']
        

        lH = logstash.TCPLogstashHandler(
            host    = logSpecs['logstash']['host'], 
            port    = logSpecs['logstash']['port'], 
            version = logSpecs['logstash']['version'],
            tags    = tags)
        
        logger.addHandler(lH)

    # set the level of the handler
    logger.setLevel(logLevel)
