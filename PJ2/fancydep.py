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
        buff = [word[0] for word in sentence]
        stack = ['0'] # root index
        transition = [] # empty
        while buff != [] or stack != ['0']:
            features = getFeatures(stack, buff, sentence)
            valid_trans = getValidTransitions(stack, buff)
            pred_trans = getPredictedTransition(features, weights, valid_trans)
            true_trans = getTrueTransition(stack, buff, dependents_of_word)
            if true_trans == -1:
                break
            transition.append(true_trans)
            if pred_trans != true_trans:
                weights = updateWeights(weights, pred_trans, true_trans, features)
    return weights

def getFeatures(stack, buff, sentence):
    depth = 3
    sw = ['EMPTY'] * depth # word
    st = ['EMPTY_POS'] * depth # tag
    sd = ['0'] * depth # distance
    for i, pos in enumerate(list(reversed(stack))):
        if i == depth:
            break
        if pos == '0':
            sw[i] = 'ROOT'
            st[i] = 'ROOT_POS'
        else:
            word_idx = int(pos) - 1
            sw[i] = sentence[word_idx][1]
            st[i] = sentence[word_idx][3]
            sd[i] = sentence[word_idx][0]

    bw = ['EMPTY'] * depth
    bt = ['EMPTY_POS'] * depth
    bd = ['0'] * depth
    for i, pos in enumerate(buff):
        if i == depth:
            break
        if pos == '0':
            bw[i] = 'ROOT'
            bt[i] = 'ROOT_POS'
        else:
            word_idx = int(pos) - 1
            bw[i] = sentence[word_idx][1]
            bt[i] = sentence[word_idx][3]
            bd[i] = sentence[word_idx][0]
    
    # single 8
    s1wd = (sw[0], sd[0])
    s2wd = (sw[1], sd[1])
    b1wd = (bw[0], bd[0])
    b2wd = (bw[1], bd[1])
    single = sw[0:2] + st[0:2] + bw[0:2] + bt[0:2]

    # pair 6
    s1wb1w = (sw[0], bw[0])
    s1tb1t = (st[0], bt[0])
    s1ws2w = (sw[0], sw[1])
    b1wb2w = (bw[0], bw[1])
    s1ts2t = (st[0], st[1])
    b1tb2t = (bt[0], bt[1])
    pair = [s1wb1w,s1tb1t,s1ws2w,b1wb2w,s1ts2t,b1tb2t]
    
    # three 4
    s123w = tuple(sw[0:3])
    s123t = tuple(st[0:3])
    b123w = tuple(bw[0:3])
    b123t = tuple(bt[0:3])
    three = [s123w, s123t, b123w, b123t]
    
    features = single + pair + three + ['bias']
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
    a = 0
    for (sentence, dependents_of_word)  in test_data:
        buff = [word[0] for word in sentence] 
        stack = ['0']
        head_of = dict() # key: dependent, val: head
        while buff != [] or stack != ['0']:
            features = getFeatures(stack, buff, sentence)
            valid_trans = getValidTransitions(stack, buff)
            pred_trans = getPredictedTransition(features, weights, valid_trans)
            
            if valid_trans == []:
                break
            elif pred_trans == 0: # leftArc
                head_of[stack[-1]] = buff[0]
                del stack[-1]
            elif pred_trans == 1: # rightArc
                head_of[buff[0]] = stack[-1]
                buff[0] = stack[-1]
                del stack[-1]
            elif pred_trans == 2: # shift
                stack.append(buff[0])
                del buff[0]
        # log prediction results
        a = a+1
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

if __name__ == "__main__":

    dev_data = readDataset('en.dev')
    test_data = readDataset('en.tst')
    
    train_file_name = 'en.tr'
    ref_file_name = 'en.dev'
    ref_output_file_name = 'en.dev.out'
    test_file_name = 'en.tst'
    test_output_file_name = 'en.tst.fancy.out'
    

    
    train_data = readDataset(train_file_name)
#    testOracleParser(train_data,0)
#    testAllNonProj(train_data)
    nFeatures = 18
    weights = []
    for iter in range(4):
        print 'iteration =', iter
        train_data = readDataset(train_file_name)
        weights = train(train_data, weights, nFeatures)
        test(dev_data, weights, test_output_file_name)
        d.eval(ref_file_name, test_output_file_name)
        
        