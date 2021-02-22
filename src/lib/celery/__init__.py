'''library for the celery worker

This contains a library that will generate a celery app. This is
a library that is provided so that everything can be made as 
simple as possible. There is no need to change anything in this
library, and this library should work as is. The currelt celery
library works usign an updated logger, and this will create its
own logger. 

All requirements for this library can be specified within the
configuration file ``config/celery.json`` Currently this relies
upon you geenrating the broker and results backend, all of which
can be easily canged within the configuration file. 

.. code-block:: python 
   :emphasize-lines: 2,9

    {
        "base":{
            "name"        : "mammogram-cv",
            "BROKER_URL"  : "redis://localhost:6379/0",
            "BACKEND_URL" : "redis://localhost:6379/1",
            "include"     : ["lib.celeryWorkerExample.worker_1"]
        },

        "extra" : {
            "result_expires" : 3600
        }
    }

It is absolutely essential that you specify the ``"base"`` configuration. This
is where information about the name (which defaults to the name of the project),
the ``BROKER_URL`` and the ``BACKEND_URL`` must be specified. The default is a
local Redis instance, and this will certainly have to be modified to suit your
needs.

All workers must be specified in the ``base.includes`` specification. You may
specify as many as you want. 

All other information **must** be specified within the ``extra`` configuration.

Once this is specified, it is possible to run a set of celery workers using the 
command ``make runCelery`` in the ``src`` folder. This will allow you run 4
parallel workers. If you want to start many more (depending upon your processor
capabilities) you should start the celery worker yourself using the command:

.. code-block:: bash

    celery -A lib.celery.App worker --concurrency=10  --loglevel=INFO


Note that celery provides multiple ways of startng workers as shown 
[here](http://docs.celeryproject.org/en/latest/userguide/workers.html) 
including autoscaling, etc. and you are welcome to experiment with all its
features. 
'''