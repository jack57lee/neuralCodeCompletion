# xxx revise it on 01/09, add parent
# Add attn_size in input_data, data_producer; add change_yT for indicating whether to remove the location of unk(just label it as unk)
# refactor the code of contructing the long line (def padding_and_concat)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import time
from collections import Counter, defaultdict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def input_data(N_filename, T_filename):
  start_time = time.time()
  with open(N_filename, 'rb') as f:
    print ("reading data from ", N_filename)
    save = pickle.load(f)
    train_dataN = save['trainData']
    test_dataN = save['testData']
    train_dataP = save['trainParent']
    test_dataP = save['testParent']
    vocab_sizeN = save['vocab_size']
    print ('the vocab_sizeN is %d (not including the eof)' %vocab_sizeN)
    print ('the number of training data is %d' %(len(train_dataN)))
    print ('the number of test data is %d\n' %(len(test_dataN)))

  with open(T_filename, 'rb') as f:
    print ("reading data from ", T_filename)
    save = pickle.load(f)
    train_dataT = save['trainData']
    test_dataT = save['testData']
    vocab_sizeT = save['vocab_size']
    attn_size = save['attn_size']
    print ('the vocab_sizeT is %d (not including the unk and eof)' %vocab_sizeT)
    print ('the attn_size is %d' %attn_size)
    print ('the number of training data is %d' %(len(train_dataT)))
    print ('the number of test data is %d' %(len(test_dataT)))
    print ('Finish reading data and take %.2f\n'%(time.time()-start_time))

  return train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, vocab_sizeT, attn_size, train_dataP, test_dataP


def data_producer(raw_data, batch_size, num_steps, vocab_size, attn_size, change_yT=False, name=None, verbose=False):

  start_time = time.time()

  with tf.name_scope(name, "DataProducer", [raw_data, batch_size, num_steps, vocab_size]):
    (raw_dataN, raw_dataT, raw_dataP) = raw_data
    assert len(raw_dataN) == len(raw_dataT)

    (vocab_sizeN, vocab_sizeT) = vocab_size
    eof_N_id = vocab_sizeN - 1
    eof_T_id = vocab_sizeT - 1
    unk_id = vocab_sizeT - 2
    
    def padding_and_concat(data, width, pad_id):
      #the size of data: a list of list. This function will pad the data according to width
      long_line = list()
      for line in data:
        pad_len = width - (len(line) % width)
        new_line = line + [pad_id] * pad_len
        assert len(new_line) % width == 0
        long_line += new_line
      return long_line

    pad_start = time.time()
    long_lineN = padding_and_concat(raw_dataN, num_steps, pad_id=eof_N_id)
    long_lineT = padding_and_concat(raw_dataT, num_steps, pad_id=eof_T_id)
    long_lineP = padding_and_concat(raw_dataP, num_steps, pad_id=1)
    assert len(long_lineN) == len(long_lineT) 
    print('Pading three long lines and take %.2fs'%(time.time()-pad_start))

    # print statistics for long_lineT  
    if verbose:
      print('Start counting the statistics of T!!')
      verbose_start = time.time()
      cnt_T = Counter(long_lineT) 
      long_lineT_len = len(long_lineT)
      empty_cnt = cnt_T[0]
      unk_cnt = cnt_T[unk_id]
      eof_cnt = cnt_T[eof_T_id]
      l_cnt = sum(np.array(long_lineT) > eof_T_id)
      w_cnt = long_lineT_len - empty_cnt - unk_cnt - eof_cnt - l_cnt
      print('long_lineT_len: %d, empty: %.4f, unk: %.4f, location: %.4f, eof: %.4f, word (except Empty): %.4f'%
            (long_lineT_len, float(empty_cnt)/long_lineT_len, float(unk_cnt)/long_lineT_len, 
            float(l_cnt)/long_lineT_len, float(eof_cnt)/long_lineT_len, float(w_cnt)/long_lineT_len))
      print('the most common 5 of cnt_T', cnt_T.most_common(5))
      print('print verbose information and take %.2fs\n'%(time.time()-verbose_start))
    
    temp_len = len(long_lineN)
    n = temp_len // (batch_size * num_steps)
    long_lineN_truncated = np.array(long_lineN[0 : n * (batch_size * num_steps)])
    long_lineP_truncated = np.array(long_lineP[0 : n * (batch_size * num_steps)])
    long_lineT_truncated_x = np.array(long_lineT[0 : n * (batch_size * num_steps)])
    long_lineT_truncated_y = np.array(long_lineT[0 : n * (batch_size * num_steps)])

    # long_lineP_truncated[long_lineP_truncated > attn_size] = attn_size  #if the parent location is too far
    long_lineP_truncated = [long_lineN_truncated[i-j] for i,j in enumerate(long_lineP_truncated)] #only store parent N

    location_index = long_lineT_truncated_x > eof_T_id    
    long_lineT_truncated_x[location_index] = unk_id
    if change_yT:
      long_lineT_truncated_y[location_index] = unk_id

    # print('count of greater than eof', sum(long_lineT_truncated_y > eof_T_id))
    
    tf_dataN = tf.convert_to_tensor(long_lineN_truncated, name="raw_dataN", dtype=tf.int32)
    tf_dataP = tf.convert_to_tensor(long_lineP_truncated, name="raw_dataP", dtype=tf.int32)
    tf_dataT_x = tf.convert_to_tensor(long_lineT_truncated_x, name="raw_dataT_x", dtype=tf.int32)
    tf_dataT_y = tf.convert_to_tensor(long_lineT_truncated_y, name="raw_dataT_y", dtype=tf.int32)
    
    data_len = len(long_lineN_truncated)
    batch_len = data_len // batch_size
    # print ('the total data length is %d, batch_len is %d\n ' %(data_len, batch_len))
    dataN = tf.reshape(tf_dataN[0 : batch_size * batch_len], [batch_size, batch_len])
    dataP = tf.reshape(tf_dataP[0 : batch_size * batch_len], [batch_size, batch_len])
    dataT_x = tf.reshape(tf_dataT_x[0 : batch_size * batch_len], [batch_size, batch_len])
    dataT_y = tf.reshape(tf_dataT_y[0 : batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps  # how many batches to complete a epoch
    assert epoch_size > 0
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    per_start = time.time()
    xN = tf.strided_slice(dataN, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    xN.set_shape([batch_size, num_steps]) # need to assert all values in x[a,:,1] are the same
    yN = tf.strided_slice(dataN, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    yN.set_shape([batch_size, num_steps])

    xT = tf.strided_slice(dataT_x, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    xT.set_shape([batch_size, num_steps]) # need to assert all values in x[a,:,1] are the same
    yT = tf.strided_slice(dataT_y, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    yT.set_shape([batch_size, num_steps])

    xP = tf.strided_slice(dataP, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    xP.set_shape([batch_size, num_steps])

    eof_indicator = tf.equal(xN[:, num_steps - 1], tf.constant([eof_N_id]*batch_size))
    print('Finish preparing input producer and takes %.2fs' %(time.time()-start_time))
    print('Each produce data takes time %.2f\n' %(time.time()-per_start))
    return xN, yN, xT, yT, epoch_size, eof_indicator, xP

if __name__ == '__main__':
  N_filename = '../pickle_data/JS_non_terminal.pickle'
  T_filename = '../pickle_data/JS_terminal_50k_whole.pickle'

  train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size, train_dataP, valid_dataP \
                                                     = input_data(N_filename, T_filename)
  train_data = (train_dataN, train_dataT, train_dataP)
  valid_data = (valid_dataN, valid_dataT, valid_dataP)
  vocab_size = (vocab_sizeN+1, vocab_sizeT+2) # N is [w, eof], T is [w, unk, eof]

  input_dataN, targetsN, input_dataT, targetsT, epoch_size, eof_indicator, input_dataP = \
      data_producer(train_data, batch_size=128, num_steps=50, vocab_size=vocab_size, attn_size=attn_size, change_yT=False, name='train', verbose=False)
  # input_dataN1, targetsN1, input_dataT1, targetsT1, epoch_size1, eof_indicator1 = \
  #     data_producer(valid_data, batch_size=128, num_steps=50, vocab_size=vocab_size, attn_size=attn_size, change_yT=False, name='test', verbose=False)      

  labels = tf.reshape(targetsT, [-1])
  eof_id = vocab_size[1] -1
  loss_condition = tf.greater(labels, tf.constant(value=eof_id, dtype=tf.int32, shape=labels.shape))
  fetches = {
      "labels":labels,
      "loss_condition":loss_condition,}
  # sess = tf.Session()  #there is no graph to run
  # vals = sess.run(fetches)
  # labels_np = vals["labels"]
  # loss_condition_np = vals["loss_condition"]
  print('*** Done! ***')