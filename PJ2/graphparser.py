import networkx as nx

# look for "TODO" in this file to see what you should do.
#
# solutions are in depparser-solution.py, but I encourage you to work
# through it on your own first


# the first thing we do is define a Weights class that will store the
# learned weights of the parser. this is just a dictionary that maps
# strings (features) to floats (parameter values).
class Weights(dict):
    # default all unknown feature values to zero
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    # given a feature vector, compute a dot product
    def dotProduct(self, x):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[feat]
        return dot

    # given an example _and_ a true label (y is +1 or -1), update the
    # weights according to the perceptron update rule (we assume
    # you've already checked that the classification is incorrect
    def update(self, x, y):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat] += y * val

# now, let's define a simple test graph that we can use as an example
# for parsing
testGraph = nx.Graph()
testGraph.add_node(0, {'word': '*root*',   'pos': '*root*'})
testGraph.add_node(1, {'word': 'the',      'pos': 'DT'})
testGraph.add_node(2, {'word': 'hairy',    'pos': 'JJ'})
testGraph.add_node(3, {'word': 'monster',  'pos': 'NN'})
testGraph.add_node(4, {'word': 'ate',      'pos': 'VB'})
testGraph.add_node(5, {'word': 'tasty',    'pos': 'JJ'})
testGraph.add_node(6, {'word': 'little',   'pos': 'JJ'})
testGraph.add_node(7, {'word': 'children', 'pos': 'NN'})
testGraph.add_edge(1, 3, {})   # the -> monster
testGraph.add_edge(2, 3, {})   # hairy -> monster
testGraph.add_edge(3, 4, {})   # monster -> ate
testGraph.add_edge(4, 0, {})   # ate -> root
testGraph.add_edge(5, 7, {})   # tasty -> children
testGraph.add_edge(6, 7, {})   # little -> children
testGraph.add_edge(7, 4, {})   # children -> ate

# we need a function that will take a sentences, compute a fully
# connected graph, and put features on edges
def computeFullGraph(inputGraph):
    # create a new graph to return
    out = nx.Graph()

    # for each pair of words (nodes) in the input graph, create an
    # edge in the output graph, on which we write some features
    for i in inputGraph.nodes():
        for j in inputGraph.nodes():
            if j <= i: continue    # we're undirected, so skip half the edges
            f = inputGraph.node[i] # get node information for i (eg {word: blah, pos: blah})
            g = inputGraph.node[j] # get node information for j

            feats = { 'w_pair=' + f['word'] + '_' + g['word']: 1.,
                      'p_pair=' + f['pos' ] + '_' + g['pos' ]: 1.,
                      'dist=' + str(abs(i-j)): 1. }
            out.add_edge(i, j, feats)
                      
    return out

# we can see what this is doing with:
#   >>> computeWeightedGraph(testGraph)[0][5]
#   {'w_pair=*root*_tasty': 1.0, 'p_pair=*root*_JJ': 1.0, 'dist=5': 1.0, 'weight': 0.0}



# we need a function that will score the edges according to our
# current weight vector:
def computeGraphEdgeWeights(graph, weights):
    for i,j in graph.edges_iter():
        graph[i][j]['weight'] = 0.   # make sure it doesn't make its way into the dot product
        # TODO: your code here
        graph[i][j]['weight'] = something_you_need_to_compute

        
# once we have a graph with weights on the edges, we need to be able
# to make a prediction (i.e., compute the MST):
def predictWeightedGraph(graph):
    # need to negate all the edge weights because we want maximum
    # spanning tree but only have a library call for
    # minimum_spanning_tree
    for i,j in graph.edges_iter():
        graph[i][j]['weight'] = - graph[i][j]['weight']
    mst = nx.minimum_spanning_tree(graph)    # gotta love libraries :0
    # now put the weights back
    for i,j in graph.edges_iter():
        graph[i][j]['weight'] = - graph[i][j]['weight']
    return mst

# compute number of mistakes
def numMistakes(true, pred):
    err = 0.
    for i,j in pred.edges_iter():
        if true.has_edge(i,j) or true.has_edge(j,i): continue   # skip
        err += 1
    return err

# now, given a graph (which has features), a true tree and a predicted
# tree, we want to update our weights
def perceptronUpdate(weights, G, true, pred):
    # first, iterate over all the edges in the predicted tree that
    # aren't in the true tree -- hint, use weights.update
    for i,j in pred.edges_iter():
        # TODO: your code here
        pass
        
    # first, iterate over all the edges in the true tree that
    # aren't in the predicted tree -- hint, use weights.update
    for i,j in true.edges_iter():
        # TODO: your code here
        pass
    
# now we can finally put it all together to make a single update on a
# single example
def runOneExample(weights, trueGraph, quiet=False):
    # first, compute the full graph and compute edge weights
    G = computeFullGraph(trueGraph)
    computeGraphEdgeWeights(G, weights)

    # make a prediction
    predGraph = 0 # TODO

    # compute the error
    err = numMistakes(trueGraph, predGraph)

    # print the predicted tree and error
    if not quiet:
        print 'error =', err, '\tpred =',
        for i,j in predGraph.edges_iter():
            print '(', trueGraph.node[i]['word'], '<->', trueGraph.node[j]['word'], ')',
        print ''
    
    # if necessary, make an update
    # TODO

    return err


# we can run this with:
# >>> weights = Weights()
# >>> runOneExample(weights, testGraph)
# error = 6.0 	pred = ( *root* <-> the ) ( *root* <-> hairy ) ( *root* <-> monster ) ( *root* <-> ate ) ( *root* <-> tasty ) ( *root* <-> little ) ( *root* <-> children ) 
# >>> runOneExample(weights, testGraph)
# error = 2.0 	pred = ( *root* <-> the ) ( the <-> monster ) ( hairy <-> monster ) ( hairy <-> children ) ( monster <-> ate ) ( tasty <-> children ) ( little <-> children ) 
# >>> runOneExample(weights, testGraph)
# error = 0.0 	pred = ( *root* <-> ate ) ( the <-> monster ) ( hairy <-> monster ) ( monster <-> ate ) ( ate <-> children ) ( tasty <-> children ) ( little <-> children ) 
#
# as you can see, the error keeps dropping and the tree gets better
# and better

# now, if you want to play around, i've given you a lot more real
# trees (en.tr) and functions (iterCoNLL) with which to read them in!
# see how well you can do!!!
#
# you can run, for instance:
# weights = Weights()
# >>> for iteration in range(5):
# ...     totalErr = 0.
# ...     for G in iterCoNLL('en.tr100'): totalErr += runOneExample(weights, G, quiet=True)
# ...     print totalErr
# ... 
# 1891.0
# 1413.0
# 1113.0
# 799.0
# 605.0
#
# as we make more iterations over the data, the error should (roughly)
# keep dropping. this is on a small subset of the overall data, but
# perhaps you can make better features that will help!!!

def iterCoNLL(filename):
    h = open(filename, 'r')
    G = None
    nn = 0
    for l in h:
        l = l.strip()
        if l == "":
            if G != None:
                yield G
            G = None
        else:
            if G == None:
                nn = nn + 1
                G = nx.Graph()
                G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
                newGraph = False
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'word' : word,
                                 'lemma': lemma,
                                 'cpos' : cpos,
                                 'pos'  : pos,
                                 'feats': feats})
            
            G.add_edge(int(head), int(id), {}) # 'true_rel': drel, 'true_par': int(id)})

    if G != None:
        yield G
    h.close()
