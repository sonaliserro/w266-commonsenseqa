'''
Utility that uses the generated question-answer concept sentences and augments the
commonsense_qa training and validation datasets. We augment the dataset in the 
following manner,

question: question options: A: optionA B: optionB C: optionC D: optionD E: optionE

question: question options: A: optionA | sentenceA B: optionB | sentenceB  
C: optionC | sentenceC D: optionD | sentenceD E: optionE | sentenceE 
'''
import nlp
import torch

import os
import sys

from transformers import T5Tokenizer

TRAIN_FILE = 'train_data.pt'
VAL_FILE = 'valid_data.pt'    

TRAIN_SENTENCES_FILE = '../models/common_gen/3_epochs/predictions_commonsense_qa_train.txt'
VALID_SENTENCES_FILE = '../models/common_gen/3_epochs/predictions_commonsense_qa_valid.txt'
    
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def get_sentences(file, dataset):
    '''
    Read the concept sentences from the provided file.

    Aggregate into groups of 5 sentences per commonsense_qa question.
    '''
    file = open(file, 'r')
    # Strip trailing newline, period, and lowercase sentence
    concept_sentences = [line.rstrip('\n').rstrip('.').lower() for line in file]
    
    # Check that there are 5 sentences per question
    assert(len(dataset) == len(concept_sentences)/5)
               
    # Group the 5 sentences per question, by question index
    sentences = []    
    for idx in range(len(dataset)):
        sentences.append(concept_sentences[idx*5:idx*5+5])    

    return sentences
    
def format_example(example, indice, concept_sentences):
    '''
    Formats the commonsense_qa example as below,

    Input
        question: John loved to paint houses. How did he usually do it? options: A: clothes get stained | i loved getting my clothes stained and painted for my house B: with brush | i loved painting houses with a brush C: wallpaper | i loved the idea of painting wallpaper on a house D: electrical circuit | john loved painting electrical circuits in his houses E: draw | i loved the drawing and painting of the houses

    Target
        B
    '''
    options = ['%s: %s | %s' % (i, option, sentence) for i, option, sentence in zip(example['choices']['label'], example['choices']['text'], concept_sentences[indice])]
    example['input_text'] = 'question: %s  options: %s' % (example['question'], ' '.join(options))
    
    # Use the following format if you want the target to be the string answer, rather than the alphabetical choice
    #example['target_text'] = '%s: %s' % (example['answerKey'], example['choices']['text'][example['choices']['label'].index(example['answerKey'])])
    example['target_text'] = '%s' % example['answerKey']
    
    return example

def format_example_train(example, indice):
    '''
    Pass-through to the format_example method with the training concept sentences
    '''
    return format_example(example, indice, train_concept_sentences)

def format_example_valid(example, indice):
    '''
    Pass-through to the format_example method with the validation concept sentences
    '''
    return format_example(example, indice, valid_concept_sentences)

def convert_to_features(example_batch):
    '''
    Tokenize the incoming examples (batch) using the initialized tokenizer with the provided arguments for input and target max length(s).
    '''
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['input_text'], truncation = True, padding = 'max_length', max_length = 256)
    target_encodings = tokenizer.batch_encode_plus(
        example_batch['target_text'], truncation = True, padding = 'max_length', max_length = 2)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

print('Loading commonsense_qa train and valid datasets')

# Load the commonsense_qa datasets
train_dataset = nlp.load_dataset('commonsense_qa', split = nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('commonsense_qa', split = nlp.Split.VALIDATION)

# Load the commonsense_qa concept sentences
train_concept_sentences = get_sentences(TRAIN_SENTENCES_FILE, train_dataset)
valid_concept_sentences = get_sentences(VALID_SENTENCES_FILE, valid_dataset)

train_dataset = train_dataset.map(format_example_train, with_indices = True, load_from_cache_file = False)        
train_dataset = train_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

valid_dataset = valid_dataset.map(format_example_valid, with_indices = True, load_from_cache_file = False) 
valid_dataset = valid_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
train_dataset.set_format(type='torch', columns = columns)
valid_dataset.set_format(type='torch', columns = columns)   
print('Processed {:d} training examples and {:d} validation examples'.format(len(train_dataset), len(valid_dataset)))

data_dir = os.path.join('../data', 'commonsense_qa_concept_sentences')
if not os.path.exists(data_dir): os.makedirs(data_dir)
print('Saving train and validation files to {}'.format(data_dir))
torch.save(train_dataset, os.path.join(data_dir, TRAIN_FILE))
torch.save(valid_dataset, os.path.join(data_dir, VAL_FILE))
