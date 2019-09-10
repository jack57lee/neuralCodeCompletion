import json
import time

train_filename = '../json_data/programs_training.json'
test_filename = '../json_data/programs_eval.json'

def process(filename):
  with open(filename, encoding='latin-1') as lines:    
    print ('Start procesing %s !!!'%(filename))
    length = 0
    line_index = 0
    for line in lines:
      line_index += 1
      if line_index % 1000 == 0:
        print ('Processing line:', line_index)
      data = json.loads(line)
      if len(data) < 3e4:
        length += len(data[:-1])  # total number of AST nodes
    return length

if __name__ == '__main__':
  start_time = time.time()
  train_len = process(train_filename)
  test_len = process(test_filename)
  print('total_length is ', train_len + test_len)
  print('Finishing counting the length and takes %.2f'%(time.time() - start_time))