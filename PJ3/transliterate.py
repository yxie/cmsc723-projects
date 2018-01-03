import sys
from optparse import OptionParser
import codecs
import string
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time
import utils
from nltk.metrics.distance import edit_distance

import data


use_cuda = torch.cuda.is_available()

# start of sequence token
SOS_token = 0
# end of sequence token
EOS_token = 1


#################################################################################
# DEFINE ENCODER DECODER MODELS
#################################################################################

################
# An Encoder model
# a visualization of the encoer architecture can be found here:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-encoder
# the RNN cell is a Gated Recurrent Unit (GRU), which adds gates to a standard RNN cell to 
# avoid vanishing/exploding gradients. 
################
class EncoderRNN(nn.Module):
    # http://pytorch.org/docs/master/nn.html
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # input_size: 85, hidden_size: 256
        # <class 'torch.nn.modules.sparse.Embedding'>
        # Embedding(num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # <class 'torch.nn.modules.rnn.GRU'>
        # GRU(input_size, hidden_size, num_layers=1)
        self.gru = nn.GRU(hidden_size, hidden_size)

    # Note that we only have to define the forward function. It 
    # will be used to construct the computation graph dynamically
    # for each new example. The backward function is automatically
    # defined for us by autograd.
    def forward(self, input, hidden):
        # embedded shape [1, 256], after reshape [1, 1, 256]
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

################
# Decoder models
################

# Simple RNN
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1) 
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result        

# RNN with attention and dropout
# as illustrated here http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
class AttnDecoderRNN(nn.Module):
    # 256, 31, 1
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=20):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output) 
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        

#################################################################################
# TRAINING
#################################################################################

# helper functions to prepare data: to train, for each pair we will need an input tensor (indexes of the
# characters in the input word) and target tensor (indexes of the
# characters in the target word). We append the EOS token to both sequences.
def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]

def variableFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, input_lang, output_lang):
    input_variable = variableFromWord(input_lang, pair[0])
    output_variable = variableFromWord(output_lang, pair[1])
    return (input_variable, output_variable)


# Train the model on one example
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=20):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    teacher_forcing_ratio = 0.5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
        
            if ni == EOS_token:
                break
    
    # use autograd to backpropagate loss
    loss.backward()
    # update model parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


# Full training process
def trainIters(pairs, input_lang, output_lang, encoder, decoder, n_iters, print_every=100, plot_every=1000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # reset every print_every
    plot_loss_total = 0 # reset every print_every

    # define criterion and optimization algorithm
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs), input_lang, output_lang) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    # now proceed one iteration at a time
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # train on one example
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, float(iter) / float(n_iters) ), iter, float(iter) / float(n_iters) * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / float(plot_every)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # plot the learning curve
    utils.showPlot(plot_losses)



#################################################################################
# GENERATE TRANSLISTERATIONS
#################################################################################

# Given an encoder-decoder model, and an input word, generate its transliteration
def generate(encoder, decoder, word, max_length=20):
    # Create input variable and initialize 
    input_variable = variableFromWord(input_lang, word)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # encode input word
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    

    # initialize decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # store produced characters and attention weights
    decoded_chars = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # generate output word one character at a time
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data

        # pick character with highest score at the output layer
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        # if the EOS token is produced, stop, otherwise go to next step
        if ni == EOS_token:
            decoded_chars.append('<EOS>')
            break
        else:
            decoded_chars.append(output_lang.index2char[ni])
            
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input


    return decoded_chars, decoder_attentions[:di + 1]


# Generate outputs for n randomly selected training examples
def generateRandomly(encoder, decoder, n=5):
    for i in range(n) :
        pair = random .choice(pairs)
        print('INPUT: ',pair[0])
        print('TARGET: ',pair[1])
        output_chars, attentions = generate(encoder, decoder, pair[0])
        print ('OUTPUT: ', ''.join(output_chars))
        print('')


# Generate outputs for the given sequence of test pairs, and compute
# the total edit distance between system output and target transliteration
def generateTest(encoder, decoder, test_pairs):
    score = 0
    outputs = []
    for pair in test_pairs:
        output_chars, attentions = generate(encoder, decoder, pair[0])
        score += edit_distance(''.join(output_chars).replace('<EOS>', ''),pair[1])
        outputs.append(''.join(output_chars).replace('<EOS>',''))
    return score, outputs


#################################################################################
# PUT IT ALL TOGETHER
#################################################################################
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--training-set", dest="file_path",
                      help="path to training set file", metavar="FILE")
    parser.add_option("-v", "--validation-set", dest="val_path",
                      help="path to validation set file", metavar="FILE")
    parser.add_option("-o", "--output-file", dest="out_path",
                      help="transliteration output for words in validation set", metavar="FILE")
    parser.add_option("-n", "--n-iterations", dest="iterations",
                      help="number of training iterations", type='int')
    
    (options, args) = parser.parse_args()

    # model parameters
    hidden_size = 256

    # training hyperparameters
    learn_rate = 0.01
    n_iter = options.iterations
    
    # how verbose
    printfreq = 1000
    plotfreq = 100
    
    # STEP 1: read in and prepare training data
    input_lang, output_lang, pairs = data.prepareTrainData(options.file_path, 'en', 'bg', reverse=True)
    
    # STEP 2: define and train sequence to sequence model
    encoder = EncoderRNN(input_lang.n_chars, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        
    trainIters(pairs, input_lang, output_lang, encoder, decoder, n_iter, print_every=printfreq, plot_every=plotfreq, learning_rate=learn_rate)
    
    # STEP 3: generate transliteration output for a random sample of training examples
    print("Examples of output for a random sample of training examples")
    generateRandomly(encoder, decoder)
    
    # STEP 4: evaluate the model on unseen validation examples
    print("Evaluate on unseen data")
    test_pairs = data.prepareTestData(options.val_path, input_lang, output_lang, reverse=True)
    distance, outputs = generateTest(encoder, decoder, test_pairs)
    if len(outputs) > 0:
        print ("Average edit distance %.4f" % (float(distance) / len(outputs)))
        f = codecs.open(options.out_path, mode='w', encoding='utf-8')
        for o in outputs:
            f.write(o)
            f.write('\n')
        f.close()

