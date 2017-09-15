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
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    # Compute count(s, w): for sense s, how many times that context word w appears in the train_texts
    # Compute count(s, all_w) = sum_j' count(s, j'): for sense s, how many context words in total appearing in the train_texts
    senses = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    context_words = list(set([word for text in train_texts for word in text]))
    count_s_w = dict()
    count_s_all_w = dict()
    # Initialization
    for sense in senses:
        count_s_all_w[sense] = 0
        for word in context_words:
            count_s_w[(sense, word)] = 0
    # Start counting
    for i in range(len(train_texts)):
        text = train_texts[i]
        sense = train_labels[i]
        count_s_all_w[sense] += len(text)
        for word in context_words:
            val = text.count(word)
            count_s_w[(sense, word)] += val
    # Compute count(s): the number of texts that have sense s
    count_s = dict()
    for sense in senses:
        count_s[sense] = train_labels.count(sense)

    # Compute p(s): probability that a text will have sense s
    # Compute p(w|s): probability that context word j will appear in a text that has sense y for 'line'
    # prob_s = dict()
    # prob_w_given_s = dict()
    weight_matrix = [] # Dimention = #sense * (#context_words + 1)
    alpha = 0.1 # smoothing constant
    # alpha = 1, accuracy = 76.7%
    # alpha = 0.5, accuracy = 84.1%
    # alpha = 0.1, accuracy = 85.9%
    for sense in senses:
        weight = []
        for word in context_words:
            prob_w_given_s = float(count_s_w[(sense, word)] + alpha) / (count_s_all_w[sense] + alpha * len(context_words))
            weight.append(math.log(prob_w_given_s))
        prob_s = float(count_s[sense]) / len(train_labels)
        bias = math.log(prob_s)
        weight.append(bias)
        weight_matrix.append(weight)

    # Testing
    # Vectorize text using bag-of-words model and create a matrix
    # Dimension = #test * (#context_words + 1)
    test_text_matrix = []
    for text in test_texts:
        text_vec = [text.count(word) for word in context_words] + [1]
        test_text_matrix.append(text_vec)

    weight_matrix = np.array(weight_matrix)
    test_text_matrix = np.array(test_text_matrix)
    # Dimention = #test * #sense
    product = test_text_matrix.dot(weight_s.transpose())
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
    vocabulary = list(set([word for text in train_texts for word in text]))
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
        # Evaluate accuracy on training data
        predicted_labels = get_predicted_labels(train_text_matrix, weight_matrix, senses)
        train_score = eval(train_labels, predicted_labels)
        print 'Iteration =', iteration, 'training score = ', train_score

    # Testing: evaluate accuracy on test data
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
    """
    **Your final implementation of Part 4 with perceptron classifier**
    """
    pass

"""
Trains a perceptron model with bag of words features  + two additional features
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    """
    **Your final implementation of Part 4 with perceptron classifier**
    """
    pass

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
    context_words = ['time', 'loss', 'export']
    count_s_w = dict()
    count_s_all_w = dict()
    # Initialization
    for sense in senses:
        count_s_all_w[sense] = 0
        for word in context_words:
            count_s_w[(sense, word)] = 0
        count_s_w[(sense, 'other')] = 0
    # Start counting
    for i in range(len(train_texts)):
        text = train_texts[i]
        sense = train_labels[i]
        count_s_all_w[sense] += len(text)
        count_other = len(text)
        for word in context_words:
            val = text.count(word)
            count_other -= val
            count_s_w[(sense, word)] += val
        count_s_w[(sense, 'other')] += count_other

    # Compute count(s): the number of texts that have sense s
    count_s = dict()
    for sense in senses:
        count_s[sense] = train_labels.count(sense)
    # Solution to Part 2.1
    print "\nSolution to Part 2.1"
    pp.pprint(count_s_w)
    pp.pprint(count_s)
    pp.pprint(count_s_all_w)

    # Compute p(s): probability that a text will have sense s
    # Compute p(w|s): probability that context word j will appear in a text that has sense y for 'line'
    prob_s = dict()
    prob_w_given_s = dict()
    for sense in senses:
        prob_s[sense] = float(count_s[sense]) / len(train_labels)
        for word in context_words:
            prob_w_given_s[(sense, word)] = float(count_s_w[(sense, word)]) / count_s_all_w[sense]
        prob_w_given_s[(sense, 'other')] = float(count_s_w[(sense, 'other')]) / count_s_all_w[sense]
    # Solution to Part 2.2
    print "\nSolution to Part 2.2"
    pp.pprint(prob_s)
    pp.pprint(prob_w_given_s)

    # Verify total probability sums to 1
    """
    prob_sum = 0.0
    for sense in senses:
        for word in context_words:
            prob_sum += prob_w_given_s[(sense, word)] * prob_s[sense]
        prob_sum += prob_w_given_s[(sense, 'other')] * prob_s[sense]
    print prob_sum
    raw_input("Press to continue")
    """

    # Test first sample of dev set
    sample_text = dev_texts[0]
    sample_label = dev_labels[0]
    sample_text_vec = [sample_text.count(w) for w in context_words]
    prob_x_given_s = dict()
    prob_x = 0
    for sense in senses:
        prob_x_given_s[sense] = get_prob_text_given_sense(
            context_words,
            sample_text_vec,
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
    print prob_sum
    raw_input("Press to continue")
    """

def get_prob_text_given_sense(context_words, text, sense, prob_w_given_s):
    result = 1.0
    for word in text:
        if word in context_words:
            p = prob_w_given_s[(sense, word)]
        else:
            p = prob_w_given_s[(sense, 'other')]
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
    text0_vec = [text0.count(word) for word in vocabulary] + [1]
    text0_vec = np.array(text0_vec)
    # Weights matrix, dimension = #sense * (#words + 1)
    weight_matrix = np.zeros((len(senses), len(vocabulary)+1))
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

if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    #running the classifier
    """
    accuracy_baseline = run_baseline_classifier(train_texts, train_targets, train_labels,
                dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    print('Test accuracy for baseline classifier', accuracy_baseline)

    test_scores = run_part2_context_words(train_texts, train_targets, train_labels,
                dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
                dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print '\nSolution to Part 2.4'
    print test_scores

    run_part3_weight_change(train_texts, train_targets, train_labels, dev_texts, dev_targets,
        dev_labels, test_texts, test_targets, test_labels)
    """

    test_score = run_bow_perceptron_classifier(train_texts, train_targets,train_labels,
                    dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print '\nSolution to Part 3.3'
    print test_score
