'''Functions for accessing databases

This library contains functions that will allow you to write
high performance code for accessing various databases. Currently
it only has access to Postgres libraries.

Specifications of the locations of the databases are assumed to be present
within the ``../config/db.json`` file. A ``../config/db.template.json`` file
has been provided for templating your ``db.json`` file with this file. 

Available Database Libraries:
-----------------------------

 - Postgres: ``pgIO``
 - SQLite: ``sqLiteIO``

'''