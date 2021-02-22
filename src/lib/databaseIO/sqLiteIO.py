from logs import logDecorator as lD
import jsonref, sqlite3

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.databaseIO.sqLiteIO'

@lD.log(logBase + '.getAllData')
def getAllData(logger, query, values=None, dbName=None):
    '''query data from the database
    
    Query the data over here. If there is a problem with the data, it is going 
    to return the value of None, and log the error. Your program needs to check 
    whether  there was an error with the query by checking for a None return 
    value. Note that the location of the dataabses are assumed to be present
    within the file ``../config/db.json``.
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    query : {str}
        The query to be made to the databse
    values : {tuple or list-like}, optional
        Additional values to be passed to the query (the default is None)
    dbName : {str or None}, optional
        The name of the database to use. If this is None, the function will 
        attempt to read the name from the ``defaultDB`` item within the 
        file ``../config/db.json``. 
    
    Returns
    -------
    list or None
        A list of tuples containing the values is returned. In case
        there is an error, the error will be logged, and a None will
        be return
    '''

    vals = None
    
    try:
        db = jsonref.load(open('../config/db.json'))

        # Check whether a dbName is available
        if (dbName is None) and ('defaultDB' in db):
            dbName = db['defaultDB']

        # Check whether a dbName has been specified
        if dbName is None:
            logger.error('A database name has not been specified.')
            return None

        conn = sqlite3.connect(db[dbName]['connection'])
        cur  = conn.cursor()
    except Exception as e:
        logger.error('Unable to connect to the database')
        logger.error(str(e))
        return

    try:

        if values is None:
            cur.execute(query)
        else:
            cur.execute(query, values)

        # We assume that the data is small so we
        # can download the entire thing here ...
        # -------------------------------------------
        vals = cur.fetchall()

    except Exception as e:
        logger.error('Unable to obtain data from the database for:\n query: {}\n{values}'.format(query, values))
        logger.error(str(e))


    try:
        cur.close()
        conn.close()
    except Exception as e:
        logger.error('Unable to disconnect to the database')
        logger.error(str(e))
        return 

    return vals

@lD.log(logBase + '.getDataIterator')
def getDataIterator(logger, query, values=None, chunks=100, dbName=None):
    '''Create an iterator from a largish query
    
    This is a generator that returns values in chunks of chunksize ``chunks``.
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    query : {str}
        The query to be made to the databse
    values : {tuple or list-like}, optional
        Additional values to be passed to the query (the default 
        is None)
    chunks : {number}, optional
        This is the number of rows that the data is going to return at every call
        if __next__() to this function. (the default is 100)
    dbName : {str or None}, optional
        The name of the database to use. If this is None, the function will 
        attempt to read the name from the ``defaultDB`` item within the 
        file ``../config/db.json``. 
    
    Yields
    ------
    list of tuples
        A list of tuples from the query, with a maximum of ``chunks`` tuples returned
        at one time. 
    '''

    try:
        db = jsonref.load(open('../config/db.json'))

        # Check whether a dbName is available
        if (dbName is None) and ('defaultDB' in db):
            dbName = db['defaultDB']

        # Check whether a dbName has been specified
        if dbName is None:
            logger.error('A database name has not been specified.')
            return None

        conn = sqlite3.connect(db[dbName]['connection'])
        cur  = conn.cursor()
    except Exception as e:
        logger.error('Unable to connect to the database')
        logger.error(str(e))
        return

    try:

        if values is None:
            cur.execute(query)
        else:
            cur.execute(query, values)

        while True:
            vals = cur.fetchmany(chunks)
            if len(vals) == 0:
                break

            yield vals

    except Exception as e:
        logger.error('Unable to obtain data from the database for:\n query: {}\nvalues'.format(query, values))
        logger.error(str(e))


    try:
        conn.close()
    except Exception as e:
        logger.error('Unable to disconnect to the database')
        logger.error(str(e))
        return 

    return

@lD.log(logBase + '.getSingleDataIterator')
def getSingleDataIterator(logger, query, values=None, dbName=None):
    '''Create an iterator from a largish query
    
    This is a generator that returns values in chunks of chunksize 1.
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    query : {str}
        The query to be made to the databse
    values : {tuple or list-like}, optional
        Additional values to be passed to the query (the default 
        is None)
    dbName : {str or None}, optional
        The name of the database to use. If this is None, the function will 
        attempt to read the name from the ``defaultDB`` item within the 
        file ``../config/db.json``. 
    
    Yields
    ------
    list of tuples
        A list of tuples from the query, with a maximum of ``chunks`` tuples returned
        at one time. 
    '''

    try:
        db = jsonref.load(open('../config/db.json'))

        # Check whether a dbName is available
        if (dbName is None) and ('defaultDB' in db):
            dbName = db['defaultDB']

        # Check whether a dbName has been specified
        if dbName is None:
            logger.error('A database name has not been specified.')
            return None

        conn = sqlite3.connect(db[dbName]['connection'])
        cur  = conn.cursor()
    except Exception as e:
        logger.error('Unable to connect to the database')
        logger.error(str(e))
        return

    try:

        if values is None:
            cur.execute(query)
        else:
            cur.execute(query, values)

        while True:
            vals = cur.fetchone()
            if vals is None:
                break

            yield vals

    except Exception as e:
        logger.error('Unable to obtain data from the database for:\n query: {}\nvalues'.format(query, values))
        logger.error(str(e))


    try:
        conn.close()
    except Exception as e:
        logger.error('Unable to disconnect to the database')
        logger.error(str(e))
        return 

    return

@lD.log(logBase + '.commitData')
def commitData(logger, query, values=None, dbName=None):
    '''query data from the database
    
    Query the data over here. If there is a problem with
    the data, it is going to return the value of ``None``, and
    log the error. Your program needs to check whether 
    there was an error with the query by checking for a ``None``
    return value
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    query : {str}
        The query to be made to the databse
    values : {tuple or list-like}, optional
        Additional values to be passed to the query (the default 
        is None)
    dbName : {str or None}, optional
        The name of the database to use. If this is None, the function will 
        attempt to read the name from the ``defaultDB`` item within the 
        file ``../config/db.json``. 
    
    Returns
    -------
    True or None
        On successful completion, a ``True`` is returned. In case
        there is an error, the error will be logged, and a ``None`` will
        be returnd
    '''

    vals = True
    
    try:
        db = jsonref.load(open('../config/db.json'))

        # Check whether a dbName is available
        if (dbName is None) and ('defaultDB' in db):
            dbName = db['defaultDB']

        # Check whether a dbName has been specified
        if dbName is None:
            logger.error('A database name has not been specified.')
            return None

        conn = sqlite3.connect(db[dbName]['connection'])
        cur  = conn.cursor()
    except Exception as e:
        logger.error('Unable to connect to the database')
        logger.error(str(e))
        return None

    try:

        if values is None:
            cur.execute(query)
        else:
            cur.execute(query, values)

    except Exception as e:
        logger.error('Unable to obtain data from the database for:\n query: {}\nvalues'.format(query, values))
        logger.error(str(e))
        vals = None


    try:
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error('Unable to disconnect to the database')
        logger.error(str(e))
        return 

    return vals

@lD.log(logBase + '.commitDataList')
def commitDataList(logger, query, values, dbName=None):
    '''query data from the database
    
    Query the data over here. If there is a problem with
    the data, it is going to return the value of None, and
    log the error. Your program needs to check whether 
    there was an error with the query by checking for a ``None``
    return value
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    query : {str}
        The query to be made to the databse
    values : {tuple or list-like}, optional
        Additional values to be passed to the query (the default 
        is None)
    dbName : {str or None}, optional
        The name of the database to use. If this is None, the function will 
        attempt to read the name from the ``defaultDB`` item within the 
        file ``../config/db.json``. 
    
    Returns
    -------
    True or None
        A successful completion of this function returns a ``True``. 
        In case there is an error, the error will be logged, and a ``None`` will
        be returned
    '''

    val = True

    try:
        db = jsonref.load(open('../config/db.json'))

        # Check whether a dbName is available
        if (dbName is None) and ('defaultDB' in db):
            dbName = db['defaultDB']

        # Check whether a dbName has been specified
        if dbName is None:
            logger.error('A database name has not been specified.')
            return None

        conn = sqlite3.connect(db[dbName]['connection'])
        cur  = conn.cursor()
    except Exception as e:
        logger.error('Unable to connect to the database')
        logger.error(str(e))
        return None

    try:
        cur.executemany(query, values)
    except Exception as e:
        logger.error('Unable to execute query for:\n query: {}\nvalues'.format(query, values))
        logger.error(str(e))
        val = None

    try:
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error('Unable to disconnect to the database')
        logger.error(str(e))
        return None

    return val

