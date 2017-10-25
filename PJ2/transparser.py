import depeval as d
import pprint as pp

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


def writePredictions(logResults, file_name):
    with open(file_name, 'w') as outh:
        for line in logResults:
            outh.write(line)


def leftArc(stack, dependents_of_word):
    assert len(stack) >= 2 and stack[-2] != 0
    dependents_of_word[stack[-1]].remove(stack[-2])
    del stack[-2]
    return (stack, dependents_of_word)

def rightArc(stack, dependents_of_word):
    assert len(stack) >= 2 and stack[-1] != 0
    dependents_of_word[stack[-2]].remove(stack[-1])
    del stack[-1]
    return (stack, dependents_of_word)

def shift(stack, buff):
    assert len(buff) > 0
    stack.append(buff[0])
    del buff[0]
    return (stack ,buff)


def isHeadDependent(idx1, idx2, dependents_of_word):
    return (idx2 in dependents_of_word.get(idx1, []))

def hasNoDependent(idx, dependents_of_word):
    return dependents_of_word.get(idx, []) == []

def getFeatures(stack, buff, sentence):
    # get features
    # corner cases, stack only has root
    if stack[-1] == '0':
        stack_top_word = 'ROOT'
        stack_top_pos = 'ROOT_POS'
    else:
        stack_word_idx = int(stack[-1]) - 1 # -1 because word_index starts from 1
        stack_top_word = sentence[stack_word_idx][1]
        stack_top_pos = sentence[stack_word_idx][3]
    # corner cases, buffer is empty
    if buff == []:
        buff_head_word = 'EMPTY'
        buff_head_pos = 'EMPTY_POS'
    else:
        buff_word_idx = int(buff[0]) - 1 # -1 because word_index starts from 1
        buff_head_word = sentence[buff_word_idx][1]
        buff_head_pos = sentence[buff_word_idx][3]
    word_pair = (stack_top_word, buff_head_word)
    pos_pair = (stack_top_pos, buff_head_pos)
    features = [stack_top_word, buff_head_word, stack_top_pos, buff_head_pos,
                word_pair, pos_pair, 'bias']
    # print features
    # raw_input()
    return features

def getPredictedTransition(features, weights):
    # compute scores
    scores = [0] * len(weights)
    for i, weight in enumerate(weights):
        for f in features:
            scores[i] += weight.get(f, 0)
    # return index with max score
    return scores.index(max(scores))

def getTrueTransition(stack, buff, dependents_of_word, quiet=1):
    true_trans = -1
    if len(stack) >= 2:
        s1_idx = stack[-1]
        s2_idx = stack[-2]
        if quiet == 0:
            print 'info:'
            print s1_idx, s2_idx
            for head in dependents_of_word:
                print head, ": ", dependents_of_word[head]
        if (isHeadDependent(s1_idx, s2_idx, dependents_of_word) # head-dependent relation
            and s2_idx != 0 # and is not root
            and hasNoDependent(s2_idx, dependents_of_word)): # s2 has no dependents
            (stack, dependents_of_word) = leftArc(stack, dependents_of_word)
            true_trans = 0
        elif (isHeadDependent(s2_idx, s1_idx, dependents_of_word)
            and s1_idx != 0
            and hasNoDependent(s1_idx, dependents_of_word)):
            (stack, dependents_of_word) = rightArc(stack, dependents_of_word)
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

def updateWeights(m, weights, pred_trans, true_trans, features):
    for f in features:
        weights[pred_trans][f] = weights[pred_trans].get(f, 0) - 1
        weights[true_trans][f] = weights[true_trans].get(f, 0) + 1
    for i in range(len(weights)):
        for f in weights[i]:
            m[i][f] = m[i].get(f, 0) + weights[i][f]
    return (m, weights)

def train(train_data, weights):
    m = [dict(), dict(), dict()]
    t = 0
    # process one iteration
    for (sentence, dependents_of_word)  in train_data:
        if quiet == 0:
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
        while buff != [] or stack != ['0']: # if buff is not empty and stack has other values than root
            ## before
            if quiet == 0:
                print 'Before: '
                print 'Buffer:', buff
                print 'Stack:', stack
                print 'Transition:', transition
            ## create training data and learn
            ## get features
            features = getFeatures(stack, buff, sentence)
            ## get predicted output
            pred_trans = getPredictedTransition(features, weights)
            ## get correct output
            true_trans = getTrueTransition(stack, buff, dependents_of_word)
            if true_trans == -1:
                break
            transition.append(true_trans)
            ## after
            if quiet == 0:
                print 'After: '
                print 'Buffer:', buff
                print 'Stack:', stack
                print 'Transition:', transition
                raw_input('>>>stop here<<<\n')
            ## update weights
            if pred_trans != true_trans:
                (m, weights) = updateWeights(m, weights, pred_trans, true_trans, features)
                t += 1
    # recover weights after one iteration
    print t
    weights = [dict(), dict(), dict()]
    for i in range(len(m)):
        for feat in m[i]:
            weights[i][feat] = m[i][feat] / t
    return weights




def logPrediction(sentence, new_head, logResults):
    for word in sentence:
        head = new_head.get(word[0], '0') # if new head is not found, default 0
        word[6] = head
        logResults.append('\t'.join(word))
    logResults.append('\n')
    return logResults

def test(test_data, weights, output_file_name):
    logResults = []
    for (sentence, dependents_of_word)  in test_data:
        # process one sentence
        buff = [word[0] for word in sentence] # all word indices
        stack = ['0'] # root index
        transition = [] # empty
        new_head = dict()
        while buff != [] or stack != ['0']: # if buff is not empty and stack has other values than root
            ## create training data and learn
            ## get features
            features = getFeatures(stack, buff, sentence)
            ## get predicted output
            pred_trans = getPredictedTransition(features, weights)
            ## go to next state
            if pred_trans == 0 and len(stack) >= 2: # leftArc
                new_head[stack[-2]] = stack[-1]
                del stack[-2]
            elif pred_trans == 1 and len(stack) >= 2: # rightArc
                new_head[stack[-1]] = stack[-2]
                del stack[-1]
            elif pred_trans == 2 and len(buff) > 0: # shift
                stack.append(buff[0])
                del buff[0]
            else:
                break
        # log prediction results
        logResults = logPrediction(sentence, new_head, logResults)
    writePredictions(logResults, output_file_name)




def testOracleParser(train_data, quiet):
    i = 0
    test_idx = 31
    for (sentence, dependents_of_word)  in train_data:
        #if i == test_idx:
        #    quiet = 0
        i += 1
        if quiet == 0:
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
        while (buff != [] or stack != ['0']): # if buff is not empty and stack has other values than root
            ## before
            if quiet == 0:
                print 'Before: '
                print 'Buffer:', buff
                print 'Stack:', stack
                print 'Transition:', transition
            ## get correct output
            true_trans = getTrueTransition(stack, buff, dependents_of_word, quiet)
            if true_trans == -1: # non-projective tree
                print 'non-projective examples', train_data.index((sentence, dependents_of_word))
                break
            transition.append(true_trans)
            ## after
            if quiet == 0:
                print 'After: '
                print 'Buffer:', buff
                print 'Stack:', stack
                print 'Transition:', transition
                raw_input('>>>stop here<<<\n')





if __name__ == "__main__":
    # print something for debugging
    quiet = 1

    # read dataset => train_data, dev_data, test_data
    # data[sentence_id][word_id][field_id]
    train_data = readDataset('en.tr100')
    dev_data = readDataset('en.dev')
    ref_file_name = 'en.dev'
    output_file_name = 'en.dev.out'

    # for each sentence
    #   update configuration
    #   obtain its transition (i.e. labels) guided by oracle
    #   based on configuration - transition pairs
    #   train the model using perceptron

    # initialize weights for features
    #   identity of word at top of the stack
    #   identity of word at head of buffer
    #   coarse POS (field 4) of word at top of the stack
    #   coarse POS (field 4) of word at head of buffer
    #   pair of words at top of stack and head of buffer
    #   pair of coarse POS (field 4) at top of stack and head of buffer

    # define leftArc = 0, rightArc = 1, shift = 2

    weight_leftArc = dict()
    weight_rightArc = dict()
    weight_shift = dict()
    weights = [weight_leftArc, weight_rightArc, weight_shift]
    # train the model
    for iter in range(10):
        print 'iteration =', iter
        train_data = readDataset('en.tr100')
        weights = train(train_data, weights)
        # test the model
        test(dev_data, weights, output_file_name)
        d.eval(ref_file_name, output_file_name)


    # testOracleParser(train_data, quiet)
