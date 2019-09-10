# use word distribution and location information(pointer)
# 1-28, reverse l_logits

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import numpy as np
import tensorflow as tf

import reader_pointer as reader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

os.environ['CUDA_VISIBLE_DEVICES']='0'
outfile = 'output_pointer_parent.txt'

N_filename = '../pickle_data/JS_non_terminal.pickle'
T_filename = '../pickle_data/JS_terminal_50k_whole.pickle'

flags = tf.flags
flags.DEFINE_string("save_path", None, #'./logs/modelT0A'
                    "Model output directory.")

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, best.")
# flags.DEFINE_string("data_path", '../data/dataJS',
#                     "Where the training/test data is stored.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
logging = tf.logging

if FLAGS.model == "test":
  outfile = 'TESToutput.txt'
def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class SmallConfig(object):
  """Small config.  get best result as 0.733 """
  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 1#1
  num_steps = 50
  attn_size = 50
  hidden_sizeN = 300
  hidden_sizeT = 500
  sizeH = 800
  max_epoch = 1#8
  max_max_epoch = 8#79
  keep_prob = 1.0#1.0
  lr_decay = 0.6#0.95
  batch_size = 64#80

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 1
  num_steps = 50
  attn_size = 50
  hidden_sizeN = 300
  hidden_sizeT = 500
  sizeH = 800
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.6
  batch_size = 80


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "best":
    return BestConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.attn_size = attn_size = config.attn_size
    self.num_steps = num_steps = config.num_steps
    self.input_dataN, self.targetsN, self.input_dataT, self.targetsT, self.epoch_size, self.eof_indicator, self.input_dataP = \
            reader.data_producer(data, batch_size, num_steps, config.vocab_size, config.attn_size, change_yT=False, name=name)
    if FLAGS.model == "test":
      self.epoch_size = 16   #small epoch size for test


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    self.attn_size = attn_size = config.attn_size
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    self.sizeN = sizeN = config.hidden_sizeN
    self.sizeT = sizeT = config.hidden_sizeT
    self.size = size = config.sizeH
    (vocab_sizeN, vocab_sizeT) = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=1.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=1.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    state_variables = []
    with tf.variable_scope("myCH0"):
      for i, (state_c, state_h) in enumerate(cell.zero_state(batch_size, data_type())):
        if i > 0: tf.get_variable_scope().reuse_variables()
        myC0 = tf.get_variable("myC0", state_c.shape[1], initializer=tf.zeros_initializer())
        myH0 = tf.get_variable("myH0", state_h.shape[1], initializer=tf.zeros_initializer())
        myC0_tensor = tf.convert_to_tensor([myC0 for _ in range(batch_size)])
        myH0_tensor = tf.convert_to_tensor([myH0 for _ in range(batch_size)])
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(myC0_tensor, myH0_tensor))

    self._initial_state = state_variables

    self.eof_indicator = input_.eof_indicator

    with tf.device("/cpu:0"):
      embeddingN = tf.get_variable(
          "embeddingN", [vocab_sizeN, sizeN], dtype=data_type())
      inputsN = tf.nn.embedding_lookup(embeddingN, input_.input_dataN)
      inputsP = tf.nn.embedding_lookup(embeddingN, input_.input_dataP)
      inputsL = tf.nn.embedding_lookup(embeddingN, input_.targetsN) # target type information

    with tf.device("/cpu:0"):
      embeddingT = tf.get_variable(
          "embeddingT", [vocab_sizeT, sizeT], dtype=data_type())
      inputsT = tf.nn.embedding_lookup(embeddingT, input_.input_dataT)

    inputs = tf.concat([inputsN, inputsT], 2)
    inputsPL = tf.concat([inputsP, inputsL], 2)
    #inputs = tf.one_hot(input_.input_data, vocab_size) 
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    attentions = []
    parents = []
    alphas = []
    state = self._initial_state
    self.memory = tf.placeholder(dtype=data_type(), shape=[batch_size, num_steps, size], name="memory")
    valid_memory = self.memory[:,-attn_size:,:]
    # print ("test test test,, state shape", np.array(state).shape)
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

        cell_parent = inputsP[:, time_step, :]
        parents.append(cell_parent)

        wm = tf.get_variable("wm", [size, size], dtype=data_type())
        wh = tf.get_variable("wh", [size, size], dtype=data_type())
        wt = tf.get_variable("wt", [size, 1], dtype=data_type())
        gt = tf.tanh(tf.matmul(tf.reshape(valid_memory, [-1, size]), wm) + tf.reshape(tf.tile(tf.matmul(cell_output, wh),[1, attn_size]), [-1, size]))
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(gt, wt), [-1,attn_size])) #the size of alpha: batch_size by attn_size
        alphas.append(alpha)
        ct = tf.squeeze(tf.matmul(tf.transpose(valid_memory, [0, 2, 1]), tf.reshape(alpha, [-1, attn_size, 1])))
        attentions.append(ct)
        valid_memory = tf.concat([valid_memory[:,1:,:], tf.expand_dims(cell_output, axis=1)], axis=1)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    attention = tf.reshape(tf.stack(axis=1, values=attentions), [-1, size])
    parent = tf.reshape(tf.stack(axis=1, values=parents), [-1, sizeN])

    self.output = tf.reshape(output, [-1, num_steps, size]) #to record the memory for next batch
    wa = tf.get_variable("wa", [size*2+sizeN, size], dtype=data_type())
    nt = tf.tanh(tf.matmul(tf.concat([output, attention, parent], axis=1), wa))

    #compute w
    softmax_w = tf.get_variable("softmax_w", [size, vocab_sizeT], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_sizeT], dtype=data_type())
    w_logits = tf.matmul(nt, softmax_w) + softmax_b
    w_probs = tf.nn.softmax(w_logits)

    #compute l
    l_logits_pre = tf.reshape(tf.stack(axis=1, values=alphas), [-1, attn_size]) #the size of alpha_reshaped: batch_size*num_steps by attn_size
    l_logits = tf.reverse(l_logits_pre, axis=[1])
    # l_probs = tf.nn.softmax(l_logits)

    #compute d
    # input_reshaped = tf.reshape(inputs, [-1, size])
    d_conditioned = tf.concat([output, attention], axis=1)
    d_w = tf.get_variable("d_w1", [2*size, 1], dtype=data_type())
    d_b = tf.get_variable("d_b1", [1], dtype=data_type())
    d = tf.nn.sigmoid(tf.matmul(d_conditioned, d_w) + d_b)

    # d_conditioned = tf.concat([output, attention], axis=1)
    # d_w1 = tf.get_variable("d_w1", [2*size, size], dtype=data_type())
    # d_b1 = tf.get_variable("d_b1", [size], dtype=data_type())
    # fc1 = tf.nn.relu(tf.matmul(d_conditioned, d_w1) + d_b1)
    # d_w2 = tf.get_variable("d_w2", [size, 1], dtype=data_type())
    # d_b2 = tf.get_variable("d_b2", [1], dtype=data_type())
    # d = tf.nn.sigmoid(tf.matmul(fc1, d_w2) + d_b2)

    #concat w and l to construct f
    f_logits = tf.concat([w_logits*d, l_logits*(1-d)], axis=1)

    labels = tf.reshape(input_.targetsT, [-1])
    weights = tf.ones([batch_size * num_steps], dtype=data_type())

    #counting unk as wrong
    unk_id = vocab_sizeT - 2
    unk_tf = tf.constant(value=unk_id, dtype=tf.int32, shape=labels.shape)
    zero_weights = tf.zeros_like(labels, dtype=data_type())
    wrong_label = tf.constant(value=-1, dtype=tf.int32, shape=labels.shape)
    condition_tf = tf.equal(labels, unk_tf)
    new_weights = tf.where(condition_tf, zero_weights, weights)
    new_labels = tf.where(condition_tf, wrong_label, labels) # only for computing the accuracy, can not be used to compute the loss(cause nan error)


    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([f_logits], [labels], [new_weights])
    probs = tf.nn.softmax(f_logits)

    # condition = tf.not_equal(labels, 182)
    # non_pad_len = tf.reduce_sum(tf.cast(condition, tf.float32))
    # mask_labels = tf.where(condition, labels, tf.constant(250, shape = labels.get_shape())) #250 just do not belong to the vocab    
    # correct_prediction = tf.equal(tf.cast(tf.argmax(probs, 1), dtype = tf.int32), mask_labels)
    # self._accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) /  non_pad_len # do not count predict <pad>(182)

    correct_prediction = tf.equal(tf.cast(tf.argmax(probs, 1), dtype = tf.int32), new_labels)
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    print ('tvars', len(tvars))
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    print ('*******the length', len(grads))
    optimizer = tf.train.AdamOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def accuracy(self):
    return self._accuracy

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  accuracy_list = []
  iters = 0
  state = session.run(model.initial_state)
  # print ('at the very initial of the run_epoch\n', state[0].c)
  eof_indicator = np.ones((model.input.batch_size), dtype=bool)
  memory = np.zeros([model.input.batch_size, model.input.num_steps, model.size])
  # file_id = session.run(model.initial_file_id) #need to remove _

  fetches = {
      "cost": model.cost,
      "accuracy": model.accuracy,
      "final_state": model.final_state,
      "eof_indicator": model.eof_indicator,
      "memory":model.output,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    # current_file_id = file_id #session.run(model.file_id)
    sub_cond = np.expand_dims(eof_indicator, axis = 1)
    condition = np.repeat(sub_cond, model.size, axis = 1)
    # zero_state = np.zeros_like(condition)
    # zero_state = np.random.uniform(-0.05,0.05,condition.shape)
    zero_state = session.run(model.initial_state)

    for i, (c, h) in enumerate(model.initial_state):
      assert condition.shape == state[i].c.shape
      feed_dict[c] = np.where(condition, zero_state[i][0], state[i].c)
      feed_dict[h] = np.where(condition, zero_state[i][1], state[i].h)
    
    feed_dict[model.memory] = memory
    vals = session.run(fetches, feed_dict)
    
    cost = vals["cost"]
    accuracy = vals["accuracy"]
    eof_indicator = vals["eof_indicator"]
    state = vals["final_state"]  #use the final state as the initial state within a whole epoch
    memory = vals["memory"]

    accuracy_list.append(accuracy)
    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f accuracy: %.4f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), np.mean(accuracy_list),
             (time.time() - start_time)))
      # print ('zero_state value', zero_state[0][0])
      # print ('gradients value', session.run(model.grads))
  
  print ('this run_epoch takes time %.2f' %(time.time() - start_time))
  return np.exp(costs / iters), np.mean(accuracy_list)




def main(_):
  start_time = time.time()
  fout = open(outfile, 'a')
  print ('\n', time.asctime(time.localtime()), file=fout)
  print ('start a new experiment %s'%outfile, file=fout)
  print ('Using dataset %s and %s'%(N_filename, T_filename), file=fout)
  print ('condition on two, two layers', file=fout)

  train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size, train_dataP, valid_dataP \
   = reader.input_data(N_filename, T_filename)

  train_data = (train_dataN, train_dataT, train_dataP)
  valid_data = (valid_dataN, valid_dataT, valid_dataP)
  vocab_size = (vocab_sizeN+1, vocab_sizeT+2) # N is [w, eof], T is [w, unk, eof]

  config = get_config()
  assert attn_size == config.attn_size #make sure the attn_size used in generate terminal is the same as the configuration
  config.vocab_size = vocab_size  
  eval_config = get_config()
  eval_config.batch_size = config.batch_size * config.num_steps
  eval_config.num_steps = 1
  eval_config.vocab_size = vocab_size

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

    # with tf.name_scope("Test"):
    #   test_input = PTBInput(config=eval_config, data=valid_data, name="TestInput")
    #   with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #     mtest = PTBModel(is_training=False, config=eval_config,
    #                      input_=test_input)


    print ('total trainable variables', len(tf.trainable_variables()), '\n\n')
    max_valid = 0
    max_step = 0
    saver = tf.train.Saver()

    sv = tf.train.Supervisor(logdir=None, summary_op=None)
    with sv.managed_session() as session:

      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print (outfile, "Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

        train_perplexity, train_accuracy = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy))
        print("Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy), file=fout)

        if i > 5:
          valid_perplexity, valid_accuracy = run_epoch(session, mvalid)
          print("Epoch: %d Valid Perplexity: ~~%.3f Valid Accuracy: %.3f~" % (i + 1, valid_perplexity, valid_accuracy))
          print("Epoch: %d Valid Perplexity: ~~%.3f Valid Accuracy: %.3f~" % (i + 1, valid_perplexity, valid_accuracy), file=fout)
          if valid_accuracy > max_valid:
            max_valid = valid_accuracy
            max_step = i + 1

      # test_perplexity, test_accuracy = run_epoch(session, mtest)
      # print("\nTest Perplexity: %.3f Test Accuracy: %.3f" % (test_perplexity, test_accuracy))

      print ('max step %d, max valid %.3f' %(max_step, max_valid))
      # print ('data path is', FLAGS.data_path)
      print ('total time takes', time.time()-start_time)
      print ('max step %d, max valid %.3f' %(max_step, max_valid), file=fout)
      print ('total time takes', time.time()-start_time, file=fout)
      fout.close()

      # if FLAGS.save_path:
      #   print("Saving model to %s." % FLAGS.save_path)
      #   save_path = saver.save(session, FLAGS.save_path, write_meta_graph=False, write_state=False)


if __name__ == "__main__":
  tf.app.run()
