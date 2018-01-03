import sys
from optparse import OptionParser
import codecs
import string
import random
import math
import time
import utils
from nltk.metrics.distance import edit_distance
import numpy as np
from attentionRNN import *
import data


def generateRandomBatch(input, batch_size):
    random.shuffle(input)
    for i in range(0, len(input), batch_size):
        yield input[i:i + batch_size]


def process_batch(batch, max_len=20):
    batch_pairs, batch_len = [(var_from_word_padded(input_lang, w0, max_len),
                               var_from_word_padded(output_lang, w1, max_len), maskMat(w1, max_len))
            for (w0, w1) in batch], \
            [(len(w0) + 1, len(w1) + 1) for (w0, w1) in batch]

    input_batch, output_batch, mask = zip(*batch_pairs)
    input_batch_len, output_batch_len = zip(*batch_len)
    input_max_len = max(input_batch_len)
    output_max_len = max(output_batch_len)
    max_len = max(input_max_len, output_max_len)
    input_batch = np.array(input_batch)[:,:input_max_len]
    output_batch = np.array(output_batch)[:, :output_max_len]
    output_mask = np.array(mask)[:, :output_max_len]
    input_batch_len = np.array(input_batch_len)
    output_batch_len = np.array(output_batch_len)
    idx = np.argsort(-input_batch_len)
    return input_batch[idx, :], output_batch[idx, :], input_batch_len[idx], output_batch_len[idx], output_mask[idx,:]


def maskMat(word, l):
    l = l - len(word) if l > len(word) else 0
    mask = [True]*len(word)
    mask.append(True)
    mask.extend([False] * l)
    return mask

def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]


# padded to l+1
def var_from_word_padded(lang, word, l):
    l = l - len(word) if l > len(word) else 0
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    indexes.extend([EOS_token] * l)
    return indexes
    # result = Variable(torch.LongTensor(indexes).view(-1, 1))
    # if use_cuda:
    #     return result.cuda()
    # else:
    #     return result


#################################################################################
# PUT IT ALL TOGETHER
#################################################################################
if __name__ == "__main__":

    file_path = 'data/en_bg.train.txt'
    # model parameters
    hidden_size = 256

    # training hyperparameters
    learn_rate = 0.01
    n_epoch = 3
    batch_size = 5
    learning_rate = 0.01

    # how verbose
    printfreq = 1000
    plotfreq = 100

    # STEP 1: read in and prepare training data
    input_lang, output_lang, pairs = data.prepareTrainData(file_path, 'en', 'bg', reverse=True)
    batch_datas = generateRandomBatch(pairs, batch_size)
    tot_entries = len(pairs)
    n_batch_per_epoch = math.ceil(len(pairs)/batch_size)
    # STEP 2: define and train sequence to sequence model

    encoder = EncoderRNN(input_size=input_lang.n_chars, embed_size=256, hidden_size=256, n_layers=1, dropout=0)
    decoder = BatchAttnDecoderRNN(hidden_size=hidden_size, embed_size=hidden_size, output_size=output_lang.n_chars,
                                  n_layers=1, dropout_p=0.1)

    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    loss = 0
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset every print_every
    plot_loss_total = 0  #
    print_every = 100 // batch_size
    plot_every = 1000 // batch_size
    # TODO: start for loop over batch_data
    iter = 0
    for ei in range(n_epoch):
        for batch_data in batch_datas:
            iter = iter + 1
            # process input output data
            input_batch, output_batch, input_batch_len, output_batch_len, output_mask = process_batch(batch_data)
            target_length = max(output_batch_len)
            # not batch first
            input_batch = np.transpose(input_batch)
            output_batch = np.transpose(output_batch)
            output_mask = np.transpose(output_mask)
            input_batch_var = Variable(torch.LongTensor(input_batch) , requires_grad=False)
            output_batch_var = Variable(torch.LongTensor(output_batch), requires_grad=False)
            output_mask_var = Variable(torch.LongTensor(output_mask), requires_grad=False)
            input_batch_var = input_batch_var.cuda() if use_cuda else input_batch_var
            output_batch_var = output_batch_var.cuda() if use_cuda else output_batch_var
            output_mask_var = output_mask_var.cuda() if use_cuda else output_mask_var
            criterion = nn.NLLLoss()

            # h0 = encoder.initHidden(batch_size)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # hn (2, 5, 256)
            # input_seq:(num_step(T), batch_size(B))
            encoder_output, hn = encoder(input_batch_var, input_batch_len)
            decoder_hidden = hn[0] + hn[1]
            decoder_hidden = decoder_hidden.unsqueeze(0)

            decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            teacher_forcing_ratio = 0.5

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_output)
                    # decoder_output (5, 31), target (5)
                    m = output_mask_var[di, :]
                    mf = m.unsqueeze(1).repeat(1, output_lang.n_chars).type(torch.FloatTensor)
                    mf = mf.cuda() if use_cuda else mf
                    # m = output_mask[di,:]
                    # o = output_batch_var[di,:]
                    loss += criterion(decoder_output * mf, output_batch_var[di, :] * m)
                    decoder_input = output_batch_var[di, :]

            else:
                m = decoder_input == SOS_token
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_output)
                    topv, topi = torch.topk(decoder_output, k=1, dim=1)
                    decoder_input = topi.squeeze()
                    # decoder_output (5, 31), target (5)
                    target = output_batch_var[di, :]
                    m = m.type(torch.LongTensor)
                    m = m.cuda() if use_cuda else m
                    mf = m.unsqueeze(1).repeat(1, output_lang.n_chars).type(torch.FloatTensor)
                    mf = mf.cuda() if use_cuda else mf

                    loss += criterion(decoder_output * mf, target*m)
                    m = decoder_input != 1

            # use autograd to backpropagate loss
            loss.backward(retain_graph=True)
            # update model parameters
            encoder_optimizer.step()
            decoder_optimizer.step()
            loss_data = loss.data[0] / sum(output_batch_len)

            print_loss_total += loss_data
            plot_loss_total += loss_data

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (utils.timeSince(start, float(iter) / float(n_batch_per_epoch*n_epoch)), iter, float(iter) / float(n_batch_per_epoch*n_epoch) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / float(plot_every)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        print_loss_total += loss
        plot_loss_total += loss

    utils.showPlot(plot_losses)




    # =============================================================================================================

    pass
