import nlp
import torch

import os
import sys
import json

import transformers
from transformers import T5Tokenizer

TRAIN_FILE = 'train_data.pt'
VAL_FILE = 'valid_data.pt'    

#SOCIAL_I_QA_LABEL_LOOKUP = {'1':'A', '2':'B', '3':'C'}
DATASET_NAMES = ['commonsense_qa', 'social_i_qa', 'common_gen']

# Read the arguments from the data_utils_args.json command line file
arg_file = sys.argv[1] if len(sys.argv) == 2 and sys.argv[1].endswith('.json') else 'data_utils_args.json'
with open(os.path.abspath(arg_file)) as json_data:
    arguments = json.load(json_data)

# Check if a valid task_name was provided
if arguments['dataset_name'] not in DATASET_NAMES:
    raise ValueError (f'Please enter a valid dataset_name, options are ({DATASET_NAMES}).')
    
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(arguments['tokenizer_name_or_path'])

# Process the examples in input and target text format for the commonsense_qa dataset
# Note that the eos token is added by t5 tokenizer.
def format_example_commonsense_qa(example):
    options = ['%s: %s' % (i, option) for i, option in zip(example['choices']['label'], example['choices']['text'])]
    example['input_text'] = 'question: %s  options: %s' % (example['question'], ' '.join(options))
    
    # Use the following format if you want the target to be the string answer, rather than the alphabetical choice
    example['target_text'] = '%s: %s' % (example['answerKey'], example['choices']['text'][example['choices']['label'].index(example['answerKey'])])
    #example['target_text'] = '%s' % example['answerKey']
    
    return example

# Process the examples in input and target text format for the social_i_qa dataset
# Note that the eos token is added by t5 tokenizer.
def format_example_social_i_qa(example):
    optionA = '%s: %s' % ('A', example['answerA'])
    optionB = '%s: %s' % ('B', example['answerB'])
    optionC = '%s: %s' % ('C', example['answerC'])
    options = ' '.join([optionA, optionB, optionC])
    example['input_text'] = 'question: %s context: %s options: %s' % (example['question'], example['context'], options)
    
    # Use the following format if you want the target to be the string answer, rather than the alphabetical choice
    label = example['label'].replace('\n', '')
    example['target_text'] = '%s' % optionA if label == '1' else optionB if label == '2' else optionC
    #example['target_text'] = '%s' % SOCIAL_I_QA_LABEL_LOOKUP.get(example['label'].replace('\n', ''))
    
    return example

# Process the examples in input and target text format for the common_gen dataset
# Note that the eos token is added by t5 tokenizer.
def format_example_common_gen(example):
    example['input_text'] = 'generate sentence: %s' % ' '.join(example['concepts'])
    example['target_text'] = '%s' % example['target']

    return example

# Wrapper format_method to handle the different task(s).
def format_example(example):
    if arguments['dataset_name'] == 'commonsense_qa':
        return format_example_commonsense_qa(example)
    elif arguments['dataset_name'] == 'social_i_qa':
        return format_example_social_i_qa(example)
    elif arguments['dataset_name'] == 'common_gen':
        return format_example_common_gen(example)
        
# Tokenize the examples, using the supplied padding arguments
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['input_text'], truncation = True, padding = 'max_length', max_length = arguments['max_len'])
    target_encodings = tokenizer.batch_encode_plus(
        example_batch['target_text'], truncation = True, padding = 'max_length', max_length = arguments['target_max_len'])

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

print('Getting data from nlp datasets')
train_dataset = nlp.load_dataset(arguments['dataset_name'], split = nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset(arguments['dataset_name'], split = nlp.Split.VALIDATION)
        
train_dataset = train_dataset.map(format_example, load_from_cache_file = False)
train_dataset = train_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

valid_dataset = valid_dataset.map(format_example, load_from_cache_file = False)
valid_dataset = valid_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
train_dataset.set_format(type='torch', columns = columns)
valid_dataset.set_format(type='torch', columns = columns)   
print('Processed {:d} training examples and {:d} validation examples'.format(len(train_dataset), len(valid_dataset)))

data_dir = os.path.join('./data', arguments['dataset_name'])
if not os.path.exists(data_dir): os.makedirs(data_dir)
print('Saving train and validation files to {}'.format(data_dir))
torch.save(train_dataset, os.path.join(data_dir, TRAIN_FILE))
torch.save(valid_dataset, os.path.join(data_dir, VAL_FILE))
