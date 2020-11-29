'''
Utility to download and format datasets for the T5-base model, using the
following sequence, based on arguments provided in the `data_utils_args.json`
file,

1. Download the train and validation dataset(s) from the huggingface 
datasets library, https://github.com/huggingface/datasets/tree/master/datasets

2. Format each example using the `dataset.map` utility into a T5 compatible 
text format. For multiple-choice datasets, we use the following format,

Input -> question: question options: A: choiceA B: choiceB C: choiceC
Target -> A: choiceA

Refer to dataset-specifc methods for exact input/target format.

3. Tokenize each example using the `dataset.map` utility using the T5-base
tokenizer. Input/Target max length, and truncation/padding features are read
from the supplied arguments. Input/Target attention masks are included as part
of each example.

4. Save the resulting files in PyTorch format to disk.
'''
import nlp
import torch

import os
import sys
import json

import transformers
from transformers import T5Tokenizer

TRAIN_FILE = 'train_data.pt'
VAL_FILE = 'valid_data.pt'    

DATA_DIR = './data'

#SOCIAL_I_QA_LABEL_LOOKUP = {'1':'A', '2':'B', '3':'C'}
HELLASWAG_LABEL_LOOKUP = {'0':'A', '1':'B', '2':'C', '3':'D'}

DATASET_NAMES = ['commonsense_qa', 'social_i_qa', 'common_gen', 'cosmos_qa', 'hellaswag']

# Read the arguments from the data_utils_args.json command line file
arg_file = sys.argv[1] if len(sys.argv) == 2 and sys.argv[1].endswith('.json') else 'data_utils_args.json'
with open(os.path.abspath(arg_file)) as json_data:
    arguments = json.load(json_data)

# Check if a valid task_name was provided
if arguments['dataset_name'] not in DATASET_NAMES:
    raise ValueError (f'Please enter a valid dataset_name, options are ({DATASET_NAMES}).')
    
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(arguments['tokenizer_name_or_path'])

def format_example_commonsense_qa(example):
    '''
    Formats the commonsense_qa example as below,
    
    Input
        question: She was always helping at the senior center, it brought her what? 
        options: A: satisfaction B: heart C: feel better D: pay E: happiness
     
    Target        
        E: happiness
    '''
    options = ['%s: %s' % (i, option) for i, option in zip(example['choices']['label'], example['choices']['text'])]
    example['input_text'] = 'question: %s  options: %s' % (example['question'], ' '.join(options))
    
    # Use the following format if you want the target to be the string answer, rather than the alphabetical choice
    example['target_text'] = '%s: %s' % (example['answerKey'], example['choices']['text'][example['choices']['label'].index(example['answerKey'])])
    #example['target_text'] = '%s' % example['answerKey']
    
    return example

def format_example_social_i_qa(example):
    '''
    Formats the social_i_qa example as below,
    
    Input
        question: How would you describe Sydney? context: Sydney walked past a homeless woman 
        asking for change but did not have any money they could give to her. Sydney felt bad 
        afterwards. options: A: sympathetic B: like a person who was unable to help C: incredulous
    Target 
        A: sympathetic
    '''
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

def format_example_common_gen(example):
    '''
    Formats the common_gen example as below,
    
    Input
        generate sentence: ski mountain skier
    
    Target
        A skier is skiing down a mountain.
    '''
    example['input_text'] = 'generate sentence: %s' % ' '.join(example['concepts'])
    example['target_text'] = '%s' % example['target']

    return example

def format_example_cosmos_qa(example):
    '''    
    Formats the cosmos_qa example as below,
    
    Input
        question:  What's a possible reason the writer needed someone to dress him every morning? 
        context: It's a very humbling experience when you need someone to dress you every morning, 
        tie your shoes, and put your hair up. Every menial task takes an unprecedented amount of effort. 
        It made me appreciate Dan even more. But anyway I shan't dwell on this (I'm not dying after all) 
        and not let it detract from my lovely 5 days with my friends visiting from Jersey. 
        options: A: The writer doesn't like putting effort into these tasks. 
        B: The writer has a physical disability. 
        C: The writer is bad at doing his own hair. D: None of the above choices.
        
    Target
        B: The writer has a physical disability.    
    '''
    option_dict = {   0: '%s: %s' % ('A', example['answer0'])
                    , 1: '%s: %s' % ('B', example['answer1'])
                    , 2: '%s: %s' % ('C', example['answer2'])
                    , 3: '%s: %s' % ('D', example['answer3'])}
    options = " ".join(option_dict.values())

    example['input_text'] = 'question: %s context: %s options: %s' % (example['question'], example['context'], options)

    label = example['label']
    example['target_text'] = option_dict[label]
    
    return example

def format_example_hellaswag(example):
    '''
    Formats the hellaswag example as below,
    
    Input
        question: A boy is bent over in his bedroom. He is trying to put on a shoe. he 
        context: Putting on shoes options: A: sits down, playing with a broken lace. 
        B: walks down the hall and picks up his shoe. C: then leans over, looks at it, 
        and throws it to the ground. D: does it over and over again.
       
    Target
        A: sits down, playing with a broken lace.
    '''
    options = ['%s: %s' % (i, option) for i, option in zip(HELLASWAG_LABEL_LOOKUP.values(), example['endings'])]
    example['input_text'] = 'question: %s context: %s options: %s' % (
        example['ctx'], example['activity_label'], ' '.join(options))

    example['target_text'] = '%s: %s' % (HELLASWAG_LABEL_LOOKUP.get(example['label']),
                                         example['endings'][int(example['label'])])

    return example

def format_example(example):
    '''
    Forwards the format_example method to the correct implementation for the dataset.
    '''
    if arguments['dataset_name'] == 'commonsense_qa':
        return format_example_commonsense_qa(example)
    elif arguments['dataset_name'] == 'social_i_qa':
        return format_example_social_i_qa(example)
    elif arguments['dataset_name'] == 'common_gen':
        return format_example_common_gen(example)
    elif arguments['dataset_name'] == 'cosmos_qa':
        return format_example_cosmos_qa(example)
    elif arguments['dataset_name'] == 'hellaswag':
        return format_example_hellaswag(example)
        
def convert_to_features(example_batch):
    '''
    Tokenize the incoming examples (batch) using the initilized tokenizer with the provided arguments for input and target max length(s).
    '''
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

print('Getting data from huggingface datasets')
# Use the following to load only a percentage of data for sample efficiency tests
train_dataset = nlp.load_dataset(arguments['dataset_name'], split = 'train[:60%]')
valid_dataset = nlp.load_dataset(arguments['dataset_name'], split = 'validation[:100%]')

#train_dataset = nlp.load_dataset(arguments['dataset_name'], split = nlp.Split.TRAIN)
#valid_dataset = nlp.load_dataset(arguments['dataset_name'], split = nlp.Split.VALIDATION)
      
train_dataset = train_dataset.map(format_example, load_from_cache_file = False)
train_dataset = train_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

valid_dataset = valid_dataset.map(format_example, load_from_cache_file = False)
valid_dataset = valid_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
train_dataset.set_format(type='torch', columns = columns)
valid_dataset.set_format(type='torch', columns = columns)   
print('Processed {:d} training examples and {:d} validation examples'.format(len(train_dataset), len(valid_dataset)))

data_dir = os.path.join(DATA_DIR, arguments['dataset_name'])
if not os.path.exists(data_dir): os.makedirs(data_dir)
print('Saving train and validation files to {}'.format(data_dir))
torch.save(train_dataset, os.path.join(data_dir, TRAIN_FILE))
torch.save(valid_dataset, os.path.join(data_dir, VAL_FILE))
