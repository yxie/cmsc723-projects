import depeval as d
import pprint as pp

# data is a list of tuples (sentence, dependency) one per sentence
# sentence is a list of words and words is a list of word properties
# dependency is a dictionary key->list of deps
def readDataset(file_name):
    data = []
    sentence = []
    dependents_of_word = dict()

    with open(file_name) as lines:
        for line in lines:
            if line == '\n':
                data.append((sentence, dependents_of_word))
                sentence = []
                dependents_of_word = dict()
            else:
                word_fields = line.split('\t')
                sentence.append(word_fields)
                head = word_fields[6]
                dependent = word_fields[0]
                if head in dependents_of_word:
                    dependents_of_word[head].append(dependent)
                else:
                    dependents_of_word[head] = [dependent]
    return data

def initializeWeights(n):
    n = n + 1 # nFeatures + bias
    weight_leftArc =  [dict() for x in range(n)]
    weight_rightArc = [dict() for x in range(n)]
    weight_shift = [dict() for x in range(n)]
    weights = [weight_leftArc, weight_rightArc, weight_shift]
    return weights

def train(train_data, weights, nFeatures):
    if weights == []:
        weights = initializeWeights(nFeatures)
    for (sentence, dependents_of_word)  in train_data:
        head_of = getHeadOf(dependents_of_word)
        dow = dependents_of_word.copy()
        buff = [word[0] for word in sentence]
        stack = ['0'] # root index
        transition = [] # empty
        while buff != [] or stack != ['0']:
            features = getFeatures(stack, buff, sentence,
                                   dow, head_of)
            valid_trans = getValidTransitions(stack, buff)
            pred_trans = getPredictedTransition(features, weights, valid_trans)
            true_trans = getTrueTransition(stack, buff, dependents_of_word)
            if true_trans == -1:
                break
            transition.append(true_trans)
            if pred_trans != true_trans:
                weights = updateWeights(weights, pred_trans, true_trans, features)
    return weights

def getFeatures(stack, buff, sentence, dependents_of_word, head_of):
    depth = 2
    sw = ['EMPTY'] * depth # word
    sp = ['EMPTY_POS'] * depth # tag
    sd = ['0'] * depth # distance
    s0h = '-1' # head of s0
    s0r = '-1' # right most dependent of s0
    s0l = '-1' # left most
    s0vr = '0' # right valency
    s0vl = '0' # left valency
    for i, pos in enumerate(list(reversed(stack))):
        if i == depth:
            break
        if i == 0:
            s0h = head_of.get(pos, '-1')
            sdep = dependents_of_word.get(pos, [])
            sdep = [int(n) for n in sdep]
            sdepr = [j1 for j1 in sdep if j1 > int(pos)]
            sdepl = [j1 for j1 in sdep if j1 < int(pos)]
            if sdepl != []:
                s0l = str(min(sdepl))
                s0vl = str(len(sdepl))
            if sdepr != []:
                s0r = str(max(sdepr))
                s0vr = str(len(sdepr))
        if pos == '0':
            sw[i] = 'ROOT'
            sp[i] = 'ROOT_POS'
        else:
            word_idx = int(pos) - 1
            sw[i] = sentence[word_idx][1]
            sp[i] = sentence[word_idx][3]
            sd[i] = sentence[word_idx][0]

    nw = ['EMPTY'] * depth
    np = ['EMPTY_POS'] * depth
    nd = ['0'] * depth
    n0h = '-1'
    n0r = '-1'
    n0l = '-1'
    n0vr = '0'
    n0vl = '0'
    for i, pos in enumerate(buff):
        if i == depth:
            break
        if i == 0:
            n0h = head_of.get(pos, '-1')
            ndep = dependents_of_word.get(pos, [])
            ndep = [int(n) for n in ndep]
            ndepr = [j1 for j1 in ndep if j1 > int(pos)]
            ndepl = [j1 for j1 in ndep if j1 < int(pos)]
            if ndepl != []:
                n0l = str(min(ndepl))
                n0vl = str(len(ndepl))
            if ndepr != []:
                n0r = str(max(ndepr))
                n0vr = str(len(ndepr))
        if pos == '0':
            nw[i] = 'ROOT'
            np[i] = 'ROOT_POS'
        else:
            word_idx = int(pos) - 1
            nw[i] = sentence[word_idx][1]
            np[i] = sentence[word_idx][3]
            nd[i] = sentence[word_idx][0]
    
    # single world basic 12
    s0wp = (sw[0], sp[0])
    n0wp = (nw[0], np[0])
    s1wp = (sw[1], sp[1])
    n1wp = (nw[1], np[1])
    single = sw[0:2] + sp[0:2] + nw[0:2] + np[0:2] # 8
#    + [s0wp,s1wp,n0wp,n1wp] # 12

    # word pair basic 8
    s0wn0w = (sw[0], nw[0])
    s0pn0p = (sp[0], np[0])
    s0ws1w = (sw[0], sw[1])
    n0wn1w = (nw[0], nw[1])
    s0ps1p = (sp[0], sp[1])
    n0pn1p = (np[0], np[1])
    
    n0pn1p = (np[0], np[1])
    s0wpn0wp = s0wp + n0wp
    s0wpn0w = s0wp + (nw[0],)
    s0wn0wp = (sw[0],) + n0wp
    s0wpn0p = s0wp + (np[0],)
    s0pn0wp = (sp[0],) + n0wp
    pair = [s0wn0w,s0pn0p,s0ws1w,n0wn1w,s0ps1p,n0pn1p] # 6
#    pair = [s0wn0w,s0pn0p,n0pn1p,s0wpn0wp,s0wpn0w,s0wn0wp,s0wpn0p,s0pn0wp]
    
    # three words 6
    n012p = tuple(np[0:3])
    s0n01p = tuple(sp[0:1] + np[0:2])
    s0hp = getProp(sentence, s0h, 3)
    s0lp = getProp(sentence, s0l, 3)
    s0rp = getProp(sentence, s0r, 3)
    n0lp = getProp(sentence, n0l, 3)
    s0hps0pn0p = (s0hp, sp[0], np[0])
    s0ps0lpn0p = (sp[0], s0lp, np[0])
    s0ps0rpn0p = (sp[0], s0rp, np[0])
    s0pn0pn0lp = (sp[0], np[0], n0lp)
    triplet = [n012p,s0n01p,s0hps0pn0p,s0ps0lpn0p,s0ps0rpn0p,s0pn0pn0lp]
        
    s0wd = (sw[0], sd[0])
    s1wd = (sw[1], sd[1])
    n0wd = (nw[0], nd[0])
    n1wd = (nw[1], nd[1])
    s0wn0w_d = (sw[0] , nw[0], str( int(sd[0]) - int(nd[0]) ) )
    s0pn0p_d = (sp[0] , np[0], str( int(sd[0]) - int(nd[0]) ) )
    dist = [s0wd, s1wd, n0wd, n1wd, s0wn0w_d, s0pn0p_d] # 6
    
    s0wvr = (sw[0], s0vr)
    s0pvr = (sp[0], s0vr)
    s0wvl = (sw[0], s0vl)
    s0pvl = (sp[0], s0vl)
    n0wvl = (nw[0], n0vl)
    n0pvl = (np[0], n0vl)
    valency = [s0wvr,s0pvr,s0wvl,s0pvl,n0wvl,n0pvl] # 6
    
    features = single + pair + dist + valency + triplet +  ['bias']
    return features

# 0: leftA, 1: rightA, 2: shift 
def getValidTransitions(stack, buff):
    valid_trans = []
    if len(stack) >= 1 and len(buff) >= 1:
        if stack[-1] != '0':
            valid_trans.append(0)
        valid_trans.append(1)
    if len(buff) >= 1:
        valid_trans.append(2)
    return valid_trans

def getPredictedTransition(features, weights, valid_trans):
    scores = [0] * len(weights)
    for i in range(len(weights)):
        if i in valid_trans:
            for f_id, f_val in enumerate(features):
                scores[i] += weights[i][f_id].get(f_val, 0)
        else:
            scores[i] = -1000000000
    # return index with max score
    return scores.index(max(scores))

# parse sentense to get transation
def getTrueTransition(stack, buff, dependents_of_word):
    true_trans = -1
    if len(stack) >= 1 and len(buff) >= 1:
        s1_idx = stack[-1]
        b1_idx = buff[0]
        if (isHeadDependent(b1_idx, s1_idx, dependents_of_word)
            and s1_idx != '0' # and is not root
            and hasNoDependent(s1_idx, dependents_of_word)):
            (stack, buff, dependents_of_word) = \
            leftArc(stack, buff, dependents_of_word)
            true_trans = 0
        elif (isHeadDependent(s1_idx, b1_idx, dependents_of_word)
            and hasNoDependent(b1_idx, dependents_of_word)):
            (stack, buff, dependents_of_word) = \
            rightArc(stack, buff, dependents_of_word)
            true_trans = 1
        elif len(buff) > 0:
            (stack, buff) = shift(stack, buff)
            true_trans = 2
        else:
            true_trans = -1 # non-projective
    elif len(buff) > 0:
        (stack, buff) = shift(stack, buff)
        true_trans = 2
    else:
        true_trans = -1 # non-projective
    # assert true_trans != -1
    return true_trans

# true if idx1 is head of idx2
def isHeadDependent(idx1, idx2, dependents_of_word):
    return (idx2 in dependents_of_word.get(idx1, []))

def hasNoDependent(idx, dependents_of_word):
    return dependents_of_word.get(idx, []) == []

def leftArc(stack, buff, dependents_of_word):
    dependents_of_word[buff[0]].remove(stack[-1])
    del stack[-1]
    return (stack, buff, dependents_of_word)

def rightArc(stack, buff, dependents_of_word):
    dependents_of_word[stack[-1]].remove(buff[0])
    buff[0] = stack[-1]
    del stack[-1]
    return (stack, buff, dependents_of_word)

def shift(stack, buff):
    stack.append(buff[0])
    del buff[0]
    return (stack ,buff)

def updateWeights(weights, pred_trans, true_trans, features):
    for f_id, f_val in enumerate(features):
        weights[pred_trans][f_id][f_val] = \
        weights[pred_trans][f_id].get(f_val, 0) - 1
        
        weights[true_trans][f_id][f_val] = \
        weights[true_trans][f_id].get(f_val, 0) + 1
    return (weights)

def testOracleParser(train_data, idx):
    (sentence, dependents_of_word) = train_data[idx]
    print 'word info:'
    for word in sentence:
        print word
    print 'head-dependent info:'
    for head in dependents_of_word:
        print head, ": ", dependents_of_word[head]
    raw_input('\n')

    # process one sentence
    buff = [word[0] for word in sentence] # all word indices
    stack = ['0'] # root index
    transition = [] # empty
    true_trans = 0
    
    while (buff != [] or stack != ['0']):
        print 'before: ', list(reversed(stack)), ',', buff
#        print 'Transition:', transition
        true_trans = getTrueTransition(stack, buff, dependents_of_word)
        if true_trans == -1: # non-projective tree
            print 'non-projective examples', train_data.index((sentence, dependents_of_word))
            break
        transition.append(true_trans)
        print 'trans: ', true_trans
        if  buff == [] and stack == []:
            print 'after: [],[]'
        elif buff == [] and stack != []:
            print 'after: ', list(reversed(stack)), ',[]'
        elif buff != [] and stack == []:
            print 'after: []', ',', buff
        else:
            print 'after: ', list(reversed(stack)), ',', buff
        raw_input('>>>stop here<<<\n')
    print 'Transition:', transition

def testAllNonProj(train_data):
    i = 0
    for (sentence, dependents_of_word)  in train_data:
        i += 1
        # process one sentence
        buff = [word[0] for word in sentence] # all word indices
        stack = ['0'] # root index
        transition = [] # empty
        true_trans = 0
        while (buff != [] or stack != ['0']):
            true_trans = getTrueTransition(stack, buff, dependents_of_word)
            if true_trans == -1: # non-projective tree
                print 'non-projective examples', train_data.index((sentence, dependents_of_word))
                break
            transition.append(true_trans)

def test(test_data, weights, output_file_name):
    logResults = []
    for (sentence, _)  in test_data:
        dependents_of_word = dict()
        buff = [word[0] for word in sentence] 
        stack = ['0']
        head_of = dict() # key: dependent, val: head
        while buff != [] or stack != ['0']:
            features = getFeatures(stack, buff, sentence,
                                   dependents_of_word, head_of)
            valid_trans = getValidTransitions(stack, buff)
            pred_trans = getPredictedTransition(features, weights, valid_trans)
            
            if valid_trans == []:
                break
            elif pred_trans == 0: # leftArc
                if buff[0] in dependents_of_word:
                    dependents_of_word[buff[0]].append(stack[-1])
                else:
                    dependents_of_word[buff[0]] = [stack[-1]]
                head_of[stack[-1]] = buff[0]
                del stack[-1]
            elif pred_trans == 1: # rightArc
                if stack[-1] in dependents_of_word:
                    dependents_of_word[stack[-1]].append(buff[0])
                else:
                    dependents_of_word[stack[-1]] = [buff[0]]
                head_of[buff[0]] = stack[-1]
                buff[0] = stack[-1]
                del stack[-1]
            elif pred_trans == 2: # shift
                stack.append(buff[0])
                del buff[0]
        # log prediction results
        logResults = logPrediction(sentence, head_of, logResults)
    writePredictions(logResults, output_file_name)

def logPrediction(sentence, head_of, logResults):
    for word in sentence:
        head = head_of.get(word[0], '0') # if new head is not found, default 0
        word[6] = head
        logResults.append('\t'.join(word))
    logResults.append('\n')
    return logResults

def writePredictions(logResults, file_name):
    with open(file_name, 'w') as outh:
        for line in logResults:
            outh.write(line)
            
def analizeWeights(weights):
    a = []
    for trans, weightT in enumerate(weights):
#        a.append([dict() for x in range(len(weightT))])
        d = []
        for f, weightF in enumerate(weightT):
            v = weightF.values()
            d.append(max(v) - min(v))
#            a[trans][f][f] = [max(v), min(v)]
        a.append(d)
    return a

def getHeadOf(dependents_of_word):
    head_of = dict()
    for head, deps in dependents_of_word.iteritems():
        for dep in deps:
            head_of[dep] = head
    return head_of

def getProp(sentence, word_idx, field):
    prop = 'NA'
    if word_idx == '-1':
        return prop
    elif word_idx == '0':
        if field == 1:
            prop = 'ROOT'
        elif field == 3:
            prop = 'ROOT_POS'
    else:
        prop = sentence[int(word_idx) - 1][field]
    return prop

if __name__ == "__main__":
    
    train_file_name = 'en.tr'
    dev_file_name = 'en.dev'
    test_file_name = 'en.tst'
    dev_output_file_name = 'en.dev.out'
    test_output_file_name = 'en.tst.fancy.out'

    dev_data = readDataset(dev_file_name)
    train_data = readDataset(train_file_name)
    
#    dep = train_data[0][1]
#    a = getHeadOf(train_data[0][1])
#    b = getProp(train_data[0][0], '1', 1)
#    testOracleParser(train_data,0)
#    testAllNonProj(train_data)
    
    nFeatures = 32
    weights = []
    for iter in range(5):
        print 'iteration =', iter
        train_data = readDataset(train_file_name)
        weights = train(train_data, weights, nFeatures)
        test(dev_data, weights, test_output_file_name)
        d.eval(dev_file_name, test_output_file_name)
        
    aw = analizeWeights(weights)