#Utilities for preprocess the data

import numpy as np
from six.moves import cPickle as pickle
import json
from collections import deque
import time


def read_N_pickle(filename):
  with open(filename, 'rb') as f:
    print ("Reading data from ", filename)
    save = pickle.load(f)
    train_data = save['trainData']
    test_data = save['testData']
    vocab_size = save['vocab_size']
    print ('the vocab_size is %d' %vocab_size)
    print ('the number of training data is %d' %(len(train_data)))
    print ('the number of test data is %d' %(len(test_data)))
    print ('Finish reading data!!')
    return train_data, test_data, vocab_size

def read_T_pickle(filename):
  with open(filename, 'rb') as f:
    print ("Reading data from ", filename)
    save = pickle.load(f)
    train_data = save['trainData']
    test_data = save['testData']
    vocab_size = save['vocab_size']
    attn_size = save['attn_size']
    print ('the vocab_size is %d' %vocab_size)
    print ('the attn_size is %d' %attn_size)
    print ('the number of training data is %d' %(len(train_data)))
    print ('the number of test data is %d' %(len(test_data)))
    print ('Finish reading data!!')
    return train_data, test_data, vocab_size, attn_size


def save(filename, terminal_dict, terminal_num, vocab_size, sorted_freq_dict):
  with open(filename, 'wb') as f:
    save = {'terminal_dict': terminal_dict,'terminal_num': terminal_num, 'vocab_size': vocab_size, 'sorted_freq_dict': sorted_freq_dict,}
    pickle.dump(save, f)

def change_protocol_for_N(filename):

    f = open(filename, 'rb')
    save = pickle.load(f)
    typeDict = save['typeDict']
    numType = save['numType']
    dicID = save['dicID']
    vocab_size = save['vocab_size']
    trainData = save['trainData']
    testData = save['testData']
    typeOnlyHasEmptyValue = save['typeOnlyHasEmptyValue']
    f.close()

    f = open(filename, 'wb')
    save = {
        'typeDict': typeDict,
        'numType': numType,
        'dicID': dicID,
        'vocab_size': vocab_size,
        'trainData': trainData,
        'testData': testData,
        'typeOnlyHasEmptyValue': typeOnlyHasEmptyValue,
        }
    pickle.dump(save, f, protocol=2)
    f.close()


def change_protocol_for_T(filename):
    f = open(filename, 'rb')
    save = pickle.load(f)
    terminal_dict = save['terminal_dict']
    terminal_num = save['terminal_num']
    vocab_size = save['vocab_size']
    attn_size = save['attn_size']
    trainData = save['trainData']
    testData = save['testData']
    f.close()

    f = open(target_filename, 'wb')
    save = {'terminal_dict': terminal_dict,
            'terminal_num': terminal_num,
            'vocab_size': vocab_size, 
            'attn_size': attn_size,
            'trainData': trainData, 
            'testData': testData,
            }
    pickle.dump(save, f, protocol=2)
    f.close()

if __name__ == '__main__':
    
    # train_filename = '../json_data/small_programs_training.json'
    # test_filename = '../json_data/small_programs_eval.json'
    # N_pickle_filename = '../pickle_data/JS_non_terminal.pickle'
    # T_pickle_filename = '../pickle_data/JS_terminal_1k.pickle'
    filename = '../pickle_data/PY_non_terminal.pickle'
    read_N_pickle(filename)
    # filename = '../pickle_data/JS_terminal_1k_whole.pickle'
    # change_protocol_for_T(filename, target_filename)


    # N_train_data, N_test_data, N_vocab_size = read_N_pickle(N_pickle_filename)
    # T_train_data, T_test_data, T_vocab_size, attn_size = read_T_pickle(T_pickle_filename)
    # print(len(N_train_data), len(T_train_data))

