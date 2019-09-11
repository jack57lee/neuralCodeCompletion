# neuralCodeCompletion                                                                                              
The implementation of the IJCAI 2018 paper: Code Completion with Neural Attention and Pointer Networks
## Descriptions for the directories
### code
* myModel_commented.py: a good commented example for our main model part, i.e., pointer mixture network.
* attention.py: standard attention model for predicting terminals
* attention_N.py: standard attention model for predicting non-terminals
* attention_N_parent.py: parent attention model for predicting non-terminals
* attention_parent.py: parent attention model for predicting terminals
* pointer.py: our poirnter mixture network without parent attention
* pointer_parent.py: our poirnter mixture network with parent attention
* reader_pointer.py: reader for reading dataset (with parent)
* reader_pointer_original.py: reader for reading dataset (original without parent)
* vanillaLSTM.py: vanilla LSTM

### preprocess_code
* freq_dict.py: generate the frequency dictionary for terminals
* get_non_terminal.py: process the non-terminals (utilize AST information)
* get_terminal_dict.py: get the terminal dictionary according to the vocabulary size
* get_terminal_whole.py: the final step to process the terminals (recording location and parent information)
* get_total_length.py: calculate the total length of the file 
* output.txt: some statistics for the terminals
* utils.py: some utils to process the data

## Download the dataset
This is the link for you to download the raw dataset: [JS & PY data](http://plml.ethz.ch/)
If you do not want to get your hands dirty with data preprocess, you can download the pickle data after preprocessed here: [pickle data](https://drive.google.com/open?id=1EZZuL8Rl3tatvxpIClvO_a8JD_Oid_oY)

## How to run the code
1. Download the dataset
2. Preprocess the data into pickle files and store them in a proper directory
3. Simply adjust the parameter setting inside the code file and run using python3, e.g. python3 attention.py.
