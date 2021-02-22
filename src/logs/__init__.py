'''Module containing helper classes for logging

This module contains two classes. The first one  will be used for 
generating a new logger object, and another one for uisng that logging 
object for new tasks. Each class is modeled as a decorator, that will
inject a ``logging.getLogger`` instance as a first parameter of the 
function. This function furthermore logs the starting and ending times
of the logs, as well as the time taken for the function, using the 
``time.time`` module. 

Configuration Information
=========================

Configuring the logger is done with the help of the configuration file
``config/config.json``. Specifically, the ``logging`` key identifies all
configuration associated with logging information within this file. An 
example if the ``logging`` section is shown below. Details of the different
sections will be described in the documentation that follows.

.. code-block:: python
    :emphasize-lines: 5,10,14

    "logging":{
        
        "logBase" : "mammogram-cv",
        "level"   : "INFO",
        "specs"   : {

            "file":{
                "todo"     : true,
                "logFolder": "logs" 
            },

            "stdout":{
                "todo"     : false 
            },

            "logstash":{
                "todo"     : false,
                "version"  : 1,
                "port"     : 5959,
                "host"     : "localhost"
            }

        }
    }

The ``"level"`` Segment
-----------------------

The logging module comes preconfigured to log at the ``"INFO"`` level. 
However this can be set to one of the following levels, and is mapped 
to their respective logging levels. 

 - ``'CRITICAL'`` mapped to ``logging.CRITICAL``
 - ``'ERROR'``    mapped to ``logging.ERROR``
 - ``'WARNING'``  mapped to ``logging.WARNING``
 - ``'INFO'``     mapped to ``logging.INFO``
 - ``'DEBUG'``    mapped to ``logging.DEBUG``

The ``"specs"`` Segment
-----------------------

This module comes preconfigured for a number of logging sinks. The logs can go
either to a logging file, to the stdout, or to logstash. Each section has a 
parameter ``"todo"`` that will determine whether a particular sink shall be
added to the logging handler. The other parameters for each section is described
below.

The ``"specs.file"`` Segment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This segment is used for sending the logger output directly to a file. A base folder
should be soecified within which the logging file should be generated. Each time the
program is run, a new file is generated in the form ``YYYY-MM-DD_hh-mm-ss.log``. The 
default formatting string used is: 
``"%(asctime)s - %(name)s - %(levelname)s - %(message)s"``.

The ``"specs.stdout"`` Segment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The output can potentially also be sent to the standard output if this section is turned
on using the ``doto`` key. By default, this section is turned off. The default formatting 
string used is: 
``"%(asctime)s - %(name)s - %(levelname)s - %(message)s"``.


The ``"specs.logstash"`` Segment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to use logstash as a sink. This is entirely JSON based. This uses TCP
rather than the standard UDP. For configuring the logstash server, make sure to add the 
input:

.. code-block:: python
    
    tcp {
        'port'  => '5959'
        'codec' => 'json'
    }

The input port should match the port specified in the ``config/config.json`` the config file.
If your logstash is running on a different machine, make sure that you specify the host IP
along with the port. An example output is shown:

.. code-block:: python

    {
     "@timestamp" => 2018-08-12T03:49:25.212Z,
          "level" => "ERROR",
           "type" => "logstash",
           "port" => 55195,
       "@version" => "1",
           "host" => "Sankha-desktop.local",
           "path" => "/Users/user/Documents/programming/python/test/mytests/mnop/src/lib/testLib/simpleLib.py",
        "message" => "Unable to add the two values [3] and [a]:\\nunsupported operand type(s) for +: 'int' and 'str'",
           "tags" => [],
    "logger_name" => "mnop.lib.simpleLib.simpleTestFunction",
     "stack_info" => nil
    } 

This can then be sent to elasticsearch. If you need specific things filtered, you can 
directly use the filtering capabilities of logstash to generate this information. 

'''