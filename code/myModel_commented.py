# a word w is composed of two kinds of information: type(N) and value(T), i.e., w_i = (N_i, T_i)
# task: given a sequence of words w_1 to w_(t-1), predict the next word value T_t

class my_Model(object):
  """This class is to build my lstm model, which mainly refers to The PTB model from official tensorflow example."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    self.attn_size = attn_size = config.attn_size  # attention size
    batch_size = input_.batch_size
    num_steps = input_.num_steps  # the lstm unrolling length
    self.sizeN = sizeN = config.hidden_sizeN  # embedding size of type(N)
    self.sizeT = sizeT = config.hidden_sizeT  # embedding size of value(T)
    self.size = size = config.sizeH  # hidden size of the lstm cell
    (vocab_sizeN, vocab_sizeT) = config.vocab_size  # vocabulary size of type and value

    # from line 17 to line 33: copy from official PTB model which defines an lstm cell with drop-out and multi-layers
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
    if is_training and config.keep_prob < 1:  # drop-out when training
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)  #multi-layers

    # from line 35 to line 44: set the initial hidden states, which are two trainable vectors. Processing a new sentence starts from here.
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

    self.eof_indicator = input_.eof_indicator  # indicate whether this is the end of a sentence

    with tf.device("/cpu:0"):
      embeddingN = tf.get_variable(
          "embeddingN", [vocab_sizeN, sizeN], dtype=data_type())
      inputsN = tf.nn.embedding_lookup(embeddingN, input_.input_dataN)  # input type embedding

    with tf.device("/cpu:0"):
      embeddingT = tf.get_variable(
          "embeddingT", [vocab_sizeT, sizeT], dtype=data_type())
      inputsT = tf.nn.embedding_lookup(embeddingT, input_.input_dataT)  # input value embedding

    inputs = tf.concat([inputsN, inputsT], 2) # concatenate the type and value embedding
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []  # store hidden state at each time_step
    attentions = []  # store context attention vector at each time_step
    alphas = []  # store attention scores at each time_step
    state = self._initial_state
    self.memory = tf.placeholder(dtype=data_type(), shape=[batch_size, num_steps, size], name="memory")
    valid_memory = self.memory[:,-attn_size:,:]  # previous hidden states within the attention window

    # from line 72 to line 87: build the RNN model, and calculate attention
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)  # lstm_cell update function
        outputs.append(cell_output) # store hidden state

        # calculate attention scores alpha and context vector ct
        wm = tf.get_variable("wm", [size, size], dtype=data_type())
        wh = tf.get_variable("wh", [size, size], dtype=data_type())
        wt = tf.get_variable("wt", [size, 1], dtype=data_type())
        gt = tf.tanh(tf.matmul(tf.reshape(valid_memory, [-1, size]), wm) + tf.reshape(tf.tile(tf.matmul(cell_output, wh),[1, attn_size]), [-1, size]))
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(gt, wt), [-1,attn_size])) #the size of alpha: batch_size by attn_size
        alphas.append(alpha)
        ct = tf.squeeze(tf.matmul(tf.transpose(valid_memory, [0, 2, 1]), tf.reshape(alpha, [-1, attn_size, 1])))
        attentions.append(ct)
        valid_memory = tf.concat([valid_memory[:,1:,:], tf.expand_dims(cell_output, axis=1)], axis=1) #move forward attention window

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])  # hidden states for all time_steps
    attention = tf.reshape(tf.stack(axis=1, values=attentions), [-1, size])  # context vectors for all time_steps

    self.output = tf.reshape(output, [-1, num_steps, size]) #to record the memory for next batch
    wa = tf.get_variable("wa", [size*2, size], dtype=data_type())
    nt = tf.tanh(tf.matmul(tf.concat([output, attention], axis=1), wa))

    #compute w: the word distribution within the global vocabulary
    softmax_w = tf.get_variable("softmax_w", [size, vocab_sizeT], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_sizeT], dtype=data_type())
    w_logits = tf.matmul(nt, softmax_w) + softmax_b
    w_probs = tf.nn.softmax(w_logits)  # baseline model uses this

    #compute l: reuse attention scores as the location distribution for pointer network
    l_logits_pre = tf.reshape(tf.stack(axis=1, values=alphas), [-1, attn_size]) #the size is batch_size*num_steps by attn_size
    l_logits = tf.reverse(l_logits_pre, axis=[1])

    #compute d: a switching network to balance the above two distributions, based on hidden states and context
    d_conditioned = tf.concat([output, attention], axis=1)
    d_w = tf.get_variable("d_w1", [2*size, 1], dtype=data_type())
    d_b = tf.get_variable("d_b1", [1], dtype=data_type())
    d = tf.nn.sigmoid(tf.matmul(d_conditioned, d_w) + d_b)

    #concat w and l to construct f
    f_logits = tf.concat([w_logits*d, l_logits*(1-d)], axis=1)

    labels = tf.reshape(input_.targetsT, [-1])
    weights = tf.ones([batch_size * num_steps], dtype=data_type())

    # set mask for counting unk as wrong
    unk_id = vocab_sizeT - 2
    unk_tf = tf.constant(value=unk_id, dtype=tf.int32, shape=labels.shape)
    zero_weights = tf.zeros_like(labels, dtype=data_type())
    wrong_label = tf.constant(value=-1, dtype=tf.int32, shape=labels.shape)
    condition_tf = tf.equal(labels, unk_tf)
    new_weights = tf.where(condition_tf, zero_weights, weights)
    new_labels = tf.where(condition_tf, wrong_label, labels)

    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([f_logits], [labels], [new_weights])
    probs = tf.nn.softmax(f_logits)

    correct_prediction = tf.equal(tf.cast(tf.argmax(probs, 1), dtype = tf.int32), new_labels)
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state