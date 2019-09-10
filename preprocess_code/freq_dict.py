#freq_dict: each terminal's frequency; terminal_num: a set about all the terminals.

import numpy as np
from six.moves import cPickle as pickle
import json
from collections import Counter
import time

#attention line 28: for python dataset, not exclude the last one
train_filename = '../json_data/programs_training.json'
test_filename = '../json_data/programs_eval.json'
target_filename = '../pickle_data/freq_dict_JS.pickle'

freq_dict = Counter()
terminal_num = set()
terminal_num.add('EmptY')

def process(filename):
  with open(filename, encoding='latin-1') as lines:
    print ('Start procesing %s !!!'%(filename))
    line_index = 0
    for line in lines:
      line_index += 1
      if line_index % 1000 == 0:
        print ('Processing line:', line_index)
      data = json.loads(line)
      if len(data) < 3e4:
        for i, dic in enumerate(data[:-1]):  #JS data[:-1] or PY data
          if 'value' in dic.keys():
            terminal_num.add(dic['value'])
            freq_dict[dic['value']] += 1
          else:          
            freq_dict['EmptY'] += 1

def save(filename):
  with open(filename, 'wb') as f:
    save = {'freq_dict': freq_dict,'terminal_num': terminal_num}
    pickle.dump(save, f, protocol=2)


if __name__ == '__main__':
  start_time = time.time()
  process(train_filename)
  process(test_filename)
  save(target_filename)
  print(freq_dict['EmptY'], freq_dict['Empty'], freq_dict['empty'], freq_dict['EMPTY'])
  print('Finishing generating freq_dict and takes %.2f'%(time.time() - start_time))


