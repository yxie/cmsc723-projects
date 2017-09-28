"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""


"""
 read one of train, dev, test subsets

 subset - one of train, dev, test

 output is a tuple of three lists
    labels: one of the 6 possible senses <cord, division, formation, phone, product, text >
    targets: the index within the text of the token to be disambiguated
    texts: a list of tokenized and normalized text input (note that there can be multiple sentences)

"""
import nltk
import sklearn
import math
import numpy as np
import pprint as pp
import random

def read_dataset(subset):
    labels = []
    texts = []
    targets = []
    if subset in ['train', 'dev', 'test']:
        with open('data/wsd_'+subset+'.txt') as inp_hndl:
            for example in inp_hndl:
                label, text = example.strip().split('\t')
                text = nltk.word_tokenize(text.lower().replace('" ','"'))
                if 'line' in text:
                    ambig_ix = text.index('line')
                elif 'lines' in text:
                    ambig_ix = text.index('lines')
                else:
                    ldjal
                targets.append(ambig_ix)
                labels.append(label)
                texts.append(text)
        return (labels, targets, texts)
    else:
        print '>>>> invalid input !!! <<<<<'


"""
computes f1-score of the classification accuracy

gold_labels - is a list of the gold labels
predicted_labels - is a list of the predicted labels

output is a tuple of the micro averaged score and the macro averaged score

"""
import sklearn.metrics
def eval(gold_labels, predicted_labels):
    return ( sklearn.metrics.f1_score(gold_labels, predicted_labels, average='micro'),
             sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro') )


"""
a helper method that takes a list of predictions and writes them to a file (1 prediction per line)
predictions - list of predictions (strings)
file_name - name of the output file
"""
def write_predictions(predictions, file_name):
    with open(file_name, 'w') as outh:
        for p in predictions:
            outh.write(p+'\n')

"""
Create a vocabulary based on window size
"""
def create_vocabulary(train_texts, train_targets, window_size):
    vocabulary = []
    for i in range(len(train_texts)):
        text = train_texts[i]
        target = train_targets[i]
        index_left = max(0, target - window_size)
        index_right = min(len(text) - 1, target + window_size)
        vocabulary = vocabulary + text[index_left : index_right]
    return list(set(vocabulary))

"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    # Compute count(s, w): for sense s, how many times that context word w appears in the train_texts
    # Compute count(s, all_w) = sum_j' count(s, j'): for sense s, how many context words in total appearing in the train_texts
    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    # context_words = list(set([word for text in train_texts for word in text]))
    window_size = 5
    context_words = create_vocabulary(train_texts, train_targets, window_size)

    texts = split_text(train_texts, train_labels)
    num_vocab = len(context_words)
    num_doc = len(train_labels)
    alpha = 1
    weight_matrix = []
    # Dimention = #sense * (#context_words + 1)
    for sense in senses:
        count_s_all_w = len(texts[sense])
        weight = []
        for word in context_words:
            prob_w_given_s = float(texts[sense].count(word) + alpha) / (count_s_all_w + alpha * num_vocab)
            weight.append(math.log(prob_w_given_s))
        prob_s = float(train_labels.count(sense)) / num_doc
        weight.append(math.log(prob_s))
        weight_matrix.append(weight)

    # Testing
    # Vectorize text using bag-of-words model and create a matrix
    # Dimension = #test * (#context_words + 1)
    # test_text_matrix = []
    # for text in test_texts:
    #     text_vec = [text.count(word) for word in context_words] + [1]
    #     test_text_matrix.append(text_vec)

    test_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in context_words] + [1],
        test_texts
        ))

    weight_matrix = np.array(weight_matrix)
    test_text_matrix = np.array(test_text_matrix)
    # Dimention = #test * #sense
    product = test_text_matrix.dot(weight_matrix.transpose())
    predicted_labels = []
    for i in range(len(product)):
        index = np.argmax(product[i])
        sense = senses[index]
        predicted_labels.append(sense)
    return eval(test_labels, predicted_labels)

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    # Initialization
    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    # vocabulary = list(set([word for text in train_texts for word in text]))
    window_size = 5
    vocabulary = create_vocabulary(train_texts, train_targets, window_size)

    c = zip(train_texts, train_labels, train_targets)
    random.shuffle(c)

    train_texts = [e[0] for e in c]
    train_labels = [e[1] for e in c]
    train_targets = [e[2] for e in c]


    # Vectorize texts using bag-of-words model
    train_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in vocabulary] + [1],
        train_texts
        ))
    dev_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in vocabulary] + [1],
        dev_texts
        ))
    test_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in vocabulary] + [1],
        test_texts
        ))
    # Weight matrix, dimension = #sense * (vocabulary_size + 1)
    # Random initialization
    # weight_matrix = np.random.rand(len(senses), len(vocabulary) + 1)
    # Initialized to 0
    weight_matrix = np.zeros((len(senses), len(vocabulary) + 1))

    # Training
    alpha = 1 # learning rate
    iterations = 20
    m = weight_matrix
    m_count = 0
    for iteration in range(1, iterations+1):
        # Update weights based on training data
        for i in range(len(train_labels)):
            text_vec = train_text_matrix[i]
            correct_label = train_labels[i]
            product = text_vec.dot(weight_matrix.transpose())
            predicted_label = senses[np.argmax(product)]
            if predicted_label != correct_label:
                p_index = senses.index(predicted_label)
                c_index = senses.index(correct_label)
                weight_matrix[p_index] = weight_matrix[p_index] - alpha * text_vec
                weight_matrix[c_index] = weight_matrix[c_index] + alpha * text_vec
                m = m + weight_matrix
                m_count += 1
        # Evaluate accuracy on training data
        predicted_labels = get_predicted_labels(train_text_matrix, weight_matrix, senses)
        train_score = eval(train_labels, predicted_labels)
        print 'Iteration =', iteration, 'training score (micro, macro) = ', train_score

    # Testing: evaluate accuracy on test data
    weight_matrix = m / m_count
    predicted_labels = get_predicted_labels(test_text_matrix, weight_matrix, senses)
    test_score = eval(test_labels, predicted_labels)
    return test_score

"""
Trains a naive bayes model with bag of words features  + two additional features
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):

    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    # vocabulary = list(set([word for text in train_texts for word in text]))
    window_size = 5
    vocabulary = create_vocabulary(train_texts, train_targets, window_size)
    texts = split_text(train_texts, train_labels)
    num_vocab = len(vocabulary)
    num_doc = len(train_labels)

    # new feature 1
    pos_train, pos_vocab = pos_feature(train_texts, train_targets)
    pos_splited = split_pos(pos_train, train_labels)
    # for sense in senses:
    #     while 'NN' in pos_splited[sense]: pos_splited[sense].remove('NN')
    #     while 'NNS' in pos_splited[sense]: pos_splited[sense].remove('NNS')

    # new feature 2
    posLast_train, posLast_vocab = posLast_feature(train_texts, train_targets)
    posLast_splited = split_posLast(posLast_train, train_labels)

    num_pos_vocab = len(pos_vocab)
    num_posLast_vocab = len(posLast_vocab)
    alpha = 1
    weight_matrix = []
    # Dimention = #sense * (#vocabulary + 1)
    for sense in senses:
        count_s_all_w = len(texts[sense])
        count_s_all_pos = len(pos_splited[sense])
        count_s_all_posLast = len(posLast_splited[sense])
        weight = []
        norm_w = count_s_all_w + alpha * num_vocab
        norm_pos = count_s_all_pos + alpha * num_pos_vocab
        norm_posLast = count_s_all_posLast + alpha * num_posLast_vocab
        for word in vocabulary:
            prob_w_given_s = float(texts[sense].count(word) + alpha) / norm_w
            weight.append(math.log(prob_w_given_s))
        for pos in pos_vocab:
            prob_pos_given_s = float(pos_splited[sense].count(pos) + alpha) / norm_pos
            weight.append(math.log(prob_pos_given_s))
        for posLast in posLast_vocab:
            prob_posLast_given_s = float(posLast_splited[sense].count(posLast) + alpha) / norm_posLast
            weight.append(math.log(prob_posLast_given_s))
        prob_s = float(train_labels.count(sense)) / num_doc
        weight.append(math.log(prob_s))
        weight_matrix.append(weight)

    # Testing
    # Vectorize text using bag-of-words model and create a matrix
    # Dimension = #test * (#vocabulary + 1)
    # test_text_matrix = []
    # for text in test_texts:
    #     text_vec = [text.count(word) for word in vocabulary] + [1]
    #     test_text_matrix.append(text_vec)

    test_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in vocabulary],
        test_texts
    ))

    pos_tests, _= pos_feature(test_texts, test_targets)
    test_pos_matrix = np.array(map(
        lambda pos_test: [pos_test.count(pos) for pos in pos_vocab],
        pos_tests
    ))

    posLast_tests, _= posLast_feature(test_texts, test_targets)
    test_posLast_matrix = np.array(map(
        lambda posLast_test: [posLast_test.count(posLast) for posLast in posLast_vocab] + [1],
        posLast_tests
    ))

    weight_matrix = np.array(weight_matrix)
    # print test_text_matrix.shape, test_pos_matrix.shape, len(pos_vocab)

    test_matrix = np.concatenate((test_text_matrix, test_pos_matrix, test_posLast_matrix), axis=1)
    # Dimention = #test * #sense
    product = test_matrix.dot(weight_matrix.transpose())
    predicted_labels = []
    for i in range(len(product)):
        index = np.argmax(product[i])
        sense = senses[index]
        predicted_labels.append(sense)
    return eval(test_labels, predicted_labels)


"""
Trains a perceptron model with bag of words features  + two additional features
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    # Initialization
    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    # vocabulary = list(set([word for text in train_texts for word in text]))
    window_size = 5
    vocabulary = create_vocabulary(train_texts, train_targets, window_size)

    # Vectorize texts using bag-of-words model
    train_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in vocabulary],
        train_texts
    ))

    pos_train, pos_vocab = pos_feature(train_texts, train_targets)
    train_pos_matrix = np.array(map(
        lambda pos_train_entry: [pos_train_entry.count(pos) for pos in pos_vocab] + [1],
        pos_train
    ))
    train_matrix = np.concatenate((train_text_matrix, train_pos_matrix), axis=1)

    test_text_matrix = np.array(map(
        lambda text: [text.count(word) for word in vocabulary],
        test_texts
    ))
    pos_tests, _ = pos_feature(test_texts, test_targets)
    test_pos_matrix = np.array(map(
        lambda pos_test: [pos_test.count(pos) for pos in pos_vocab] + [1],
        pos_tests
    ))
    test_matrix = np.concatenate((test_text_matrix, test_pos_matrix), axis=1)

    # Weight matrix, dimension = #sense * (vocabulary_size + 1)
    # Random initialization
    # weight_matrix = np.random.rand(len(senses), len(vocabulary) + 1)
    # Initialized to 0
    weight_matrix = np.zeros((len(senses), len(vocabulary) + len(pos_vocab) + 1))
    m = np.zeros((len(senses), len(vocabulary) + len(pos_vocab) + 1))
    # Training
    alpha = 1  # learning rate
    iterations = 20
    for iteration in range(1, iterations + 1):
        # Update weights based on training data
        for i in range(len(train_labels)):
            text_vec = train_matrix[i]
            correct_label = train_labels[i]
            product = text_vec.dot(weight_matrix.transpose())
            predicted_label = senses[np.argmax(product)]
            if predicted_label != correct_label:
                p_index = senses.index(predicted_label)
                c_index = senses.index(correct_label)
                weight_matrix[p_index] = weight_matrix[p_index] - alpha * text_vec
                weight_matrix[c_index] = weight_matrix[c_index] + alpha * text_vec
                m = m + weight_matrix
        # Evaluate accuracy on training data
        predicted_labels = get_predicted_labels(train_matrix, weight_matrix, senses)
        train_score = eval(train_labels, predicted_labels)
        # print train_score[0], ', '
        print 'Iteration =', iteration, 'training score = ', train_score
    m = m / iterations
    # Testing: evaluate accuracy on test data
    predicted_labels = get_predicted_labels(test_matrix, m, senses)
    test_score = eval(test_labels, predicted_labels)
    return test_score


"""
Part 1.1
Baseline classifier: most frequent label
"""
def run_baseline_classifier(train_texts, train_targets,train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    most_freq_label = nltk.FreqDist(train_labels).most_common(1)[0][0]
    print('Most frequent label', most_freq_label)
    predicted_labels = [most_freq_label] * len(dev_labels)
    accuracy = sklearn.metrics.accuracy_score(dev_labels, predicted_labels)
    return accuracy

"""
Part 2.1 & 2.2
"""
def run_part2_context_words(train_texts, train_targets, train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    # Compute count(s, w): for sense s, how many times that context word w appears in the train_texts
    # Compute count(s, all_w) = sum_j' count(s, j'): for sense s, how many context words in total appearing in the train_texts
    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    vocabulary = list(set([word for text in train_texts for word in text]))
    context_words = ['time', 'loss', 'export']
    count_s_w = dict()
    count_s_all_w = dict()

    # Start counting
    for i in range(len(train_texts)):
        text = train_texts[i]
        sense = train_labels[i]
        count_s_all_w[sense] = count_s_all_w.get(sense, 0) + len(text)
        for word in vocabulary:
            val = text.count(word)
            count_s_w[(sense, word)] = count_s_w.get((sense, word), 0) + val
    # Compute count(s): the number of texts that have sense s
    count_s = dict()
    for sense in senses:
        count_s[sense] = train_labels.count(sense)
    # Solution to Part 2.1
    print "\nSolution to Part 2.1"
    print "c(s):"
    for sense in senses:
        print sense, count_s[sense]
    print "c(s, w):"
    for sense in senses:
        for word in context_words:
            print sense, word, count_s_w[(sense, word)]

    # Compute p(s): probability that a text will have sense s
    # Compute p(w|s): probability that context word j will appear in a text that has sense y for 'line'
    # Compute p(s|w) = p(s, w) / p(w) = p(w|s) * p(s) / sum_j( p(w|s_j) * p(s_j) )
    prob_s = dict()
    prob_w_given_s = dict()
    prob_s_given_w = dict()
    alpha = 1 # smoothing constant
    for sense in senses:
        prob_s[sense] = float(count_s[sense]) / len(train_labels)
        for word in vocabulary:
            prob_w_given_s[(sense, word)] = (
                float(count_s_w[(sense, word)] + alpha)
                / (count_s_all_w[sense] + alpha * len(vocabulary))
                )
    prob_w = dict()
    for word in context_words:
        prob_w[word] = 0
        for sense in senses:
            prob_w[word] += prob_w_given_s[(sense, word)] * prob_s[sense]
    for sense in senses:
        for word in context_words:
            prob_s_given_w[(sense, word)] = (
                float(prob_w_given_s[(sense, word)] * prob_s[sense])
                / prob_w[word]
            )

    # Solution to Part 2.2
    print "\nSolution to Part 2.2"
    print "p(s):"
    for sense in senses:
        print sense, prob_s[sense]
    print "p(w|s):"
    for sense in senses:
        for word in context_words:
            print sense, word, prob_w_given_s[(sense, word)]
    print "p(s|w):"
    for sense in senses:
        for word in context_words:
            print sense, word, prob_s_given_w[(sense, word)]

    # Verify total probability sums to 1
    """
    for word in context_words:
        prob_sum = 0.0
        for sense in senses:
            prob_sum += prob_s_given_w[(sense, word)]
        print "prob_sum = ", prob_sum, "for word", word
    raw_input("Press to continue")
    """

    # Test first sample of dev set
    sample_text = dev_texts[0]
    sample_label = dev_labels[0]
    prob_x_given_s = dict()
    prob_x = 0
    for sense in senses:
        prob_x_given_s[sense] = get_prob_text_given_sense(
            vocabulary,
            sample_text,
            sense,
            prob_w_given_s
            )
        prob_x += prob_x_given_s[sense] * prob_s[sense]
    prob_s_given_x = dict()
    for sense in senses:
        prob_s_given_x[sense] = prob_x_given_s[sense] * prob_s[sense] / prob_x
    # Solution to Part 2.3
    print "\nSolution to Part 2.3"
    pp.pprint(prob_s_given_x)

    # Verify total probability sums to 1
    """
    prob_sum = 0
    for sense in senses:
        prob_sum += prob_s_given_x[sense]
    print "prob_sum = ", prob_sum
    raw_input("Press to continue")
    """

def get_prob_text_given_sense(vocabulary, text, sense, prob_w_given_s):
    result = 1.0
    for word in text:
        if word in vocabulary:
            p = prob_w_given_s[(sense, word)]
            result *= p
    return result

"""
Part 3.1
"""
def run_part3_weight_change(train_texts, train_targets,train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    # Part 3.1
    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    vocabulary = list(set([word for text in train_texts for word in text]))
    # text vector
    text0 = train_texts[0]
    correct_label = train_labels[0]
    # text0_vec, dimension = 1 * (#words + 1)
    text0_vec = [text0.count(word) for word in vocabulary] + [1]
    text0_vec = np.array(text0_vec)
    # Weights matrix, dimension = #sense * (#words + 1)
    weight_matrix = np.zeros((len(senses), len(vocabulary)+1))
    # product, dimension = 1 * (#sense)
    product = text0_vec.dot(weight_matrix.transpose())
    predicted_label = senses[np.argmax(product)]

    if predicted_label != correct_label:
        print 'Wrong label: ', predicted_label
        print 'Weight change for wrong label: '
        weight_change = []
        for i in range(len(vocabulary)):
            if text0_vec[i] != 0:
                weight_change.append((vocabulary[i], -text0_vec[i]))
        print weight_change
        print 'Correct label: ', correct_label
        print 'Weight change for correct label: '
        weight_change = []
        for i in range(len(vocabulary)):
            if text0_vec[i] != 0:
                weight_change.append((vocabulary[i], text0_vec[i]))
        print weight_change

def get_predicted_labels(text_matrix, weight_matrix, senses):
    predicted_labels = []
    product = text_matrix.dot(weight_matrix.transpose())
    for i in range(len(product)):
        predicted_label = senses[np.argmax(product[i])]
        predicted_labels.append(predicted_label)
    return predicted_labels


# my code. mostly useless. ----------------------------------------------------------------
# create a file for human annotation.
def write_to_file(dev_texts, filename):
    random.shuffle(dev_texts)
    with open(filename, 'w') as outh:
		for d in dev_texts:
			outh.write(' '.join(d) + '\n')

# def write_to_file(data, file_name):
#     with open(file_name, 'w') as outh:
#         for p in data:
#             outh.write(p + '\n')

# concatinate training text according to their label.
# prob(w|s) can be done by calling text[sense].count(w) / len(text[sense]
def split_text(text,label):
    texts = dict()
    for i in range(len(text)):
        texts[label[i]] = texts.get(label[i], []) + text[i]

    # with open('data/wsd_train.txt') as inp_hndl:
    #     for example in inp_hndl:
    #         label, text = example.strip().split('\t')
    #         text = nltk.word_tokenize(text.lower().replace('" ', '"'))
    #         texts[label] = texts.get(label, []) + text
    return texts

# discard pounctuation
def read_dataset_mod(subset):
	labels = []
	texts = []
	targets = []
	if subset in ['train', 'dev', 'test']:
		with open('data/wsd_'+subset+'.txt') as inp_hndl:
			for example in inp_hndl:
				label, text = example.strip().split('\t')
				tokenizer = nltk.tokenize.RegexpTokenizer(r'\w{2,}')
				text = tokenizer.tokenize(text.lower())
				if 'line' in text:
					ambig_ix = text.index('line')
				elif 'lines' in text:
					ambig_ix = text.index('lines')
				else:
					ldjal
				targets.append(ambig_ix)
				labels.append(label)
				texts.append(text)
		return (labels, targets, texts)
	else:
		print '>>>> invalid input !!! <<<<<'

def pos_feature(texts, targets):
    pos_l_l_t = [nltk.pos_tag(texts[i][targets[i]-2:targets[i]+3]) for i in range(len(texts))]
    pos = [[pos_t[1] for pos_t in pos_l_t] for pos_l_t in pos_l_l_t]
    pos_vocab = list(set.union(*map(set,pos)))
    return pos, pos_vocab

def posLast_feature(texts, targets):
    posLasts = []
    """
    for i in range(len(texts)):
        pos = 'Unknown'
        if targets[i] - 1 >= 0:
            word_pos = nltk.pos_tag([texts[i][targets[i]-1]])
            pos = word_pos[0][1]
        posLasts = posLasts + [pos]
    """
    posLasts = [nltk.pos_tag([texts[i][targets[i]-1]])[0][1] for i in range(len(texts))]
    posLast_vocab = list(set(posLasts))

    return posLasts, posLast_vocab

    # return [nltk.pos_tag(word) for text in train_texts for word in text]
def split_pos(pos, label):
    pos_splited = dict()
    for i in range(len(pos)):
        pos_splited[label[i]]=pos_splited.get(label[i],[])+pos[i]
    return pos_splited

def split_posLast(posLast, label):
    posLast_splited = dict()
    for i in range(len(posLast)):
        posLast_splited[label[i]] = posLast_splited.get(label[i], []) + [posLast[i]]
    return posLast_splited


if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    # running the classifier
    """
    accuracy_baseline = run_baseline_classifier(train_texts, train_targets, train_labels,
                dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)
    print('Test accuracy for baseline classifier', accuracy_baseline)

    run_part2_context_words(train_texts, train_targets, train_labels,
                dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print '\nSolution to Part 2.4'
    print test_scores

    run_part3_weight_change(train_texts, train_targets, train_labels, dev_texts, dev_targets,
        dev_labels, test_texts, test_targets, test_labels)

    test_score = run_bow_perceptron_classifier(train_texts, train_targets,train_labels,
                    dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print '\nSolution to Part 3.3'
    print test_score

    test_scores = run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print test_scores
    """

    test_scores = run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print test_scores
