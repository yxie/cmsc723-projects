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

def initializeWeights():
    n = 6 + 1 # nFeatures + bias
    weight_leftArc =  [dict() for x in range(n)]
    weight_rightArc = [dict() for x in range(n)]
    weight_shift = [dict() for x in range(n)]
    weights = [weight_leftArc, weight_rightArc, weight_shift]
    return weights

def train(train_data, weights):
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
                (m, weights) = updateWeights(m, weights, pred_trans, true_trans, features)
    return weights

def getFeatures(stack, buff, sentence):
    # stack only has root
    if stack[-1] == '0':
        s1w = 'ROOT'
        s1t = 'ROOT_POS'
    else:
        word_idx = int(stack[-1]) - 1 # -1 because word_index starts from 1
        s1w = sentence[word_idx][1]
        s1t = sentence[word_idx][3]

    # buffer is empty
    if buff == []:
        b1w = 'EMPTY'
        b1t = 'EMPTY_POS'
    else:
        word_idx = int(buff[0]) - 1 # -1 because word_index starts from 1
        b1w = sentence[word_idx][1]
        b1t = sentence[word_idx][3]

    s1wb1w = (s1w, b1w)
    s1tb1t = (s1t, b1t)
    features = [s1w, b1w, s1t, b1t, s1wb1w, s1tb1t, 'bias']
    return features

# 0: leftA, 1: rightA, 2: shift 
def getValidTransitions(stack, buff):
    valid_trans = []
    assert (len(stack) >= 1), "getValidTrans: stack can't be empty"
    if len(stack) >= 2 and len(buff) >= 1:
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

def getTrueTransition(stack, buff, dependents_of_word):
    true_trans = -1
    if len(stack) >= 2:
        s1_idx = stack[-1]
        b1_idx = buff[-1]
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


if __name__ == "__main__":

    dev_data = readDataset('en.dev')
    test_data = readDataset('en.tst')
    
    train_file_name = 'en.tr100'
    ref_file_name = 'en.dev'
    ref_output_file_name = 'en.dev.out'
    test_file_name = 'en.tst'
    test_output_file_name = 'en.tst.fancy.out'
    
    weights = initializeWeights()
    
    for iter in range(10):
        print 'iteration =', iter
        train_data = readDataset(train_file_name)
        weights = train(train_data, weights)
        
        
        