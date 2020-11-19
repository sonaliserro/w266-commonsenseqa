'''
Utility to generate commonsense_qa predictions to be submitted to
their leaderboard. It does the following,

(1) Load the commonsense_qa test dataset from nlp datasets, and format it
for the T5 model, but without any target fields.
(2) Invoke the t5_eval script to generate predictions for the test file
(3) Format the predictions based on,
https://github.com/allenai/aristo-leaderboard/tree/master/openbookqa/evaluator
'''

import nlp
import torch

import os
from transformers import T5Tokenizer

# Need to do this to access t5_eval in main directory
import sys
sys.path.insert(0, '..')

import requests
import json

import t5_eval

MODEL_PATH = '../models/commonsense_qa/100_percent'
DATA_DIR = '../data/commonsense_qa'
TEST_FILE = 'test_data.pt' 
LEADERBOARD_FILE = 'predictions_leaderboard.csv'

def format_example(example):
    options = ['%s: %s' % (i, option) for i, option in zip(example['choices']['label'], example['choices']['text'])]
    example['input_text'] = 'question: %s  options: %s' % (example['question'], ' '.join(options))
    
    return example

def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['input_text'], truncation = True, padding = 'max_length', max_length = 128)
   
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
    }

    return encodings

def get_predictions():
    '''
    Helper method that invokes the t5_eval script to generate predictions.
    '''
    print('Generating predictions for test file..')
    t5_eval.main(['--model_name_or_path', MODEL_PATH,
                  '--file_path', os.path.join(DATA_DIR, TEST_FILE),
                  '--tokenizer_name_or_path', MODEL_PATH, 
                  '--max_target_length', '10', 
                  '--do_predict'])  

    predictions_file = os.path.join(MODEL_PATH, 'predictions.txt')
    predictions = open(predictions_file, 'r').readlines()
    return predictions

def get_question_ids():
    '''
    Helper method that downloads the commonsense_qa test dataset from the
    tau-nlp website to scrape the question_ids
    '''
    question_ids = []
    response = requests.get('https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl')
    # Split by newline, and get rid of trailing empty last line
    data = response.text.split('\n')[:-1]  
    for id_, row in enumerate(data):
        question_ids.append(json.loads(row)["id"])
    return question_ids
    
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load the commonsense_qa test dataset from nlp datasets
test_dataset = nlp.load_dataset('commonsense_qa', split = nlp.Split.TEST)

# Process the test dataset
test_dataset = test_dataset.map(format_example, load_from_cache_file = False)
test_dataset = test_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

# Save the pytorch dataset to disk
columns = ['input_ids', 'attention_mask']
test_dataset.set_format(type = 'torch', columns = columns) 
print('Processed {:d} examples'.format(len(test_dataset)))

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
print('Saving test file to {}'.format(DATA_DIR))
torch.save(test_dataset, os.path.join(DATA_DIR, TEST_FILE))

# Load the predictions, and check length
predictions = get_predictions()
assert(len(predictions) == len(test_dataset))

# Load the question_ids, and check length
question_ids = get_question_ids()
assert(len(question_ids) == len(test_dataset))

# Write the formatted predictions
output_file = os.path.join(MODEL_PATH, LEADERBOARD_FILE)
with open(output_file, 'w') as writer:
    for question_id, prediction in zip(question_ids, predictions):
        writer.write(','.join([question_id, prediction.split(':')[0]]))
        writer.write('\n')
print('Formatted predictions file is at {}'.format(output_file))
                      