from logs import logDecorator as lD
import jsonref, os
import matplotlib.pyplot as plt # This comes before networkx
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from datetime import datetime as dt
from lib.databaseIO import pgIO

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.resultGraph.graphLib'

@lD.log(logBase + '.generateGraph')
def generateGraph(logger):
    '''generate a directed graph from the modules config
    
    generate a networkX.Graph object by reading the contents
    of the ``config/modules/`` folder. 
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    
    Returns
    -------
    networkX.Graph object
        Graph of which object is created when
    '''
    
    try:

        graph = nx.DiGraph()

        folder = '../config/modules'
        files  = [f for f in os.listdir(folder) if f.endswith('.json')]

        for f in files:
            data = jsonref.load(open(os.path.join(folder, f)))
            inp  = list(data['inputs'].keys())
            out  = list(data['outputs'].keys())
            f = f.replace('.json', '')

            graph.add_node( f, type='module', summary='' )
            
            # Add the incoming edges
            for n in inp:
                if n not in graph.nodes:
                    summary = jsonref.dumps(data['inputs'][n])
                    graph.add_node( n, 
                        type    = data['inputs'][n]['type'],
                        summary = summary)

                graph.add_edge(n, f)

            # Add the outgoing edges
            for n in out:
                if n not in graph.nodes:
                    summary = jsonref.dumps(data['outputs'][n])
                    graph.add_node( n, 
                        type    = data['outputs'][n]['type'],
                        summary = summary)

                graph.add_edge(f, n)

    except Exception as e:
        logger.error('Unable to generate the graph: {}'.format(e))

    return graph

@lD.log(logBase + '.plotGraph')
def plotGraph(logger, graph, fileName=None):
    '''plot the graph
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    graph : {networkX.Graph object}
        The graph that needs to be plotted
    fileName : {str}, optional
        name of the file where to save the graph (the default is None, which
        results in no graph being generated)
    '''

    try:
        plt.figure()

        moduleNodes = [m for m, d in graph.nodes(data=True) if ('module' == d['type'])]
        otherNodes  = [m for m, d in graph.nodes(data=True) if ('module' != d['type'])]
        lables      = {m:m for m in graph.nodes}

        pos = graphviz_layout(graph, prog='dot')
        nx.draw_networkx_nodes(graph, pos, nodelist=moduleNodes, node_color='orange', node_size=500)
        nx.draw_networkx_nodes(graph, pos, nodelist=otherNodes, node_color='cyan', node_size=500)
        nx.draw_networkx_edges(graph, pos,  arrows=True)
        nx.draw_networkx_labels(graph, pos, lables, font_size=10)

        if fileName is not None:
            plt.savefig(fileName)

        plt.close()
        print('Graph saved ...')
    except Exception as e:
        logger.error('Unable to plot the graph: {}'.format(e))

    return

@lD.log(logBase + '.generateSubGraph')
def generateSubGraph(logger, graph, keyNode):
    '''generate a subgraph that contains all prior nodes
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    graph : {networkX.Graph object}
        [description]
    keyNode : {str}
        Name of the node whose ancestors need to be geenrated.
    
    Returns
    -------
    networkX.Graph object
        graph containing the particular node and its ancistors.
    '''

    try:
        newGraph = nx.DiGraph()

        nodes = list(nx.ancestors(graph, keyNode))
        nodes.append(keyNode)
        
        for n, d in graph.nodes(data=True):
            if (n in nodes):
                newGraph.add_node( n, **d )

        for n1, n2 in graph.edges:
            if (n1 in nodes) and (n2 in nodes):
                newGraph.add_edge(n1, n2)

    except Exception as e:
        logger.error('Unable to generate the right subgraph: {}'.format(e))

    return newGraph

@lD.log(logBase + '.graphToSerialized')
def graphToSerialized(logger, graph):
    '''serializes a graph
    
    Takes a networkX.Graph object and converts it into a serialized
    set of nodes and edges. 
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    graph : {networkX.Graph object}
        A networkX graph object that is to be serialized
    
    Returns
    -------
    tuple of serialized lists
        This is a tuple of nodes and edges in a serialized format
        that can be later directly inserted into a database.
    '''

    progName = config['logging']['logBase']

    now = dt.now()

    nodes = []
    for n, d in graph.nodes(data=True):
            nodes.append([
                progName,  # program name
                now,   # current datetime 
                n,         # node name
                d['type'], # 'module', 'csv'. ... 
                d['summary']])

    edges = []
    for n1, n2 in graph.edges:
        edges.append([progName, now, n1, n2])

    return nodes, edges

@lD.log(logBase + '.serializedToGraph')
def serializedToGraph(logger, nodes, edges):
    '''deserialize a graph serialized earlier
    
    Take serialized versions of the nodes and edges which is
    produced by the function ``graphToSerialized`` and convert
    that into a normal ``networkX``.
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    nodes : {list}
        serialized versions of the nodes of a graph
    edges : {list}
        A list of edges in a serialized form
    
    Returns
    -------
    networkX.Graph object
        Takes a list of serialized nodes and edges and converts it
        into a networkX.Graph object
    '''

    graph = nx.DiGraph() 

    for _, _, n, t, s in nodes:
        graph.add_node( n, type = t, summary = s)

    for _, _, e1, e2 in edges:
        graph.add_edge( e1, e2)
        

    return graph

@lD.log(logBase + '.uploadGraph')
def uploadGraph(logger, graph, dbName=None):
    '''upload the supplied graph to a database
    
    Given a graph, this function is going to upload the graph into
    a particular database. In case a database is not specified, this
    will try to upload the data into the default database.
    
    Parameters
    ----------
    logger : {logging.logger}
        logging element 
    graph : {networkX graph}
        The graph that needs to be uploaded into the database
    dbName : {str}, optional
        The name of the database in which to upload the data into. This
        is the identifier within the ``db.json`` configuration file. (the 
        default is ``None``, which would use the default database specified
        within the same file)
    '''

    try:

        nodes, edges = graphToSerialized(graph)

        queryNodes = '''insert into graphs.nodes values %s'''
        queryEdges = '''insert into graphs.edges values %s'''

        pgIO.commitDataList(queryNodes, nodes, dbName=dbName)
        pgIO.commitDataList(queryEdges, edges, dbName=dbName)

    except Exception as e:
        logger.error()


    return

