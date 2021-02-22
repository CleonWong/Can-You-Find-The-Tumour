'''Utilities for generating graphs

This provides a set of utilities that will allow us to geenrate a
girected graph. This assumes that configuration files for all the 
modules are present in the ``config/modules/`` folder. The files
should be JSON files with the folliwing specifications:

.. code-block:: javascript

    {
        "inputs"  : {},
        "outputs" : {},
        "params"  : {}
    }

The ``inputs`` and the ``outputs`` refer to the requirements of the
module and the result of the module. Both can be empty, but in that
case, they should be represented by empty dictionaries as shown above.

All the configuration paramethers for a particular module should go
into the dictionary that is referred to by the keyword ``params``. 

An examples of what can possibly go into the ``inputs`` and ``outputs``
is as follows:

.. code-block:: javascript

    "inputs": {
        "abc1":{
            "type"        : "file-csv",
            "location"    : "../data/abc1.csv",
            "description" : "describe how the data is arranged"
        }
    },
    "outputs" : {
        "abc2":{
            "type"        : "dbTable",
            "location"    : "<dbName.schemaName.tableName>"
            "dbHost"      : "<dbHost>",
            "dbPort"      : "<dbHost>",
            "dbName"      : "<dbName>",
            "description" : "description of the table"
        },
        "abc3":{
            "type"        : "file-png",
            "location"    : "../reports/img/Fig1.png",
            "description" : "some description of the data"
        }
    },
    "params" : {}

In the above code block, the module will comprise of a single input with
the name ``abc1`` and outputs with names ``abc2`` and ``abc3``. Each of
these objects are associated with two mandatory fields: ``type`` and 
``location``. Each ``type`` will typically have a meaningful ``location``
argument associated with it.  


Example types and their corresponding location argument:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - "file-file"         : "<string containing the location>", 
  - "file-fig"          : "<string containing the location>", 
  - "file-csv"          : "<string containing the location>", 
  - "file-hdf5"         : "<string containing the location>", 
  - "file-meta"         : "<string containing the location>", 
  - "folder-checkPoint" : "<string containing the folder>", 
  - "DB-dbTable"        : "<dbName.schemaName.tableName>", 
  - "DB-dbColumn"       : "<dbName.schemaName.tableName.columnName>" 

You are welcome to generate new ``types``s. Note that anything starting with a ``file-``
represents a file within your folder structure. Anything starting with ``folder-``
represents a folder. Examples of these include checkpoints of Tensorflow models during
training, etc. Anything starting with a ``DB-`` represents a traditional database like
Postgres. 

It is particularly important to name the different inputs and outputs consistently
throughout, and this is going to help link the different parts of the graph together.

There are functions that allow graphs to be written to the database, and subsequently
retreived. It would then be possible to generate graphs from the entire set of modules.
Dependencies can then be tracked across different progeams, not just across different 
modules.


Uploading graphs to databases:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is absolutely possible that you would like to upload the graphs into dataabses. This
can be done if the current database that you are working with has the following tables:

.. code-block:: SQL

  create schema if not exists graphs;

  create table graphs.nodes (
      program_name     text,
      now              timestamp with time zone,
      node_name        text,
      node_type        text,
      summary          text
  );

  create table graphs.edges (
      program_name     text,
      now              timestamp with time zone,
      node_from        text,
      node_to          text
  );


There are functions provided that will be able to take the entire graph and upload them
directly into the databases. 


Available Graph Libraries:
--------------------------

 - ``graphLib``: General purpose libraries for constructing graphs from the module
                 configurations. 


'''