'''
Utility to generate concept sets for the commonsense_qa question-answer choice
pairs. Uses the spaCy language model to extract the nouns, verbs, and pronouns
from the question and answer text. As an example for the following question-answer
pair, we would generate the below concept set,

question: John loved to paint houses. How did he usually do it? 
choice: with brush
concept set: John loved paint houses with brush

The generated concept sets are then formatted using the t5-base tokenizer
to produce examples for predicting sentences from a t5 model fine-tuned on
common_gen.
'''
import nlp
import torch

import os
import sys

import spacy

from transformers import T5Tokenizer

TRAIN_FILE = 'train_data.pt'
VAL_FILE = 'valid_data.pt'    
    
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load the spaCy language model
spacy_nlp = spacy.load('en_core_web_lg')

def generate_concepts(batch_example):
    '''
    Generate concept sets for the commonsense_qa question-choice pairs.

    For example the following commonsense example would generate 5 concept
    sets as shown below,

    Input:
    question: John loved to paint houses. How did he usually do it? options: A: clothes get stained B: with brush C: wallpaper D: electrical circuit E: draw

    Output:
    generate sentence: John loved paint houses clothes get stained
    generate sentence: John loved paint houses with brush
    generate sentence: John loved paint houses wallpaper
    generate sentence: John loved paint houses electrical circuit
    generate sentence: John loved paint houses draw
    '''
    concepts = []
    for question, answers in zip(batch_example['question'], batch_example['choices']):
        question_concepts = []        
        # Use the nouns and verbs as concepts
        for token in spacy_nlp(question):
            if not token.is_stop and (token.pos_ in ['NOUN', 'VERB', 'PROPN']):
                question_concepts.append(token.text)
        # Grab the first 5 concepts from the question text
        question_concepts = question_concepts[:5]
        # Use all words in the answer as concepts
        for answer in answers['text']:
            concepts += ['generate sentence: %s %s' % (' '.join(question_concepts), answer)]
            
    return {'input_text': concepts}

def convert_to_features(example_batch):
    '''
    Tokenize the incoming examples (batch) using the initialized tokenizer with the provided arguments for input and target max length(s).
    '''
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['input_text'], truncation = True, padding = 'max_length', max_length = 16)
    
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask']
    }

    return encodings

print('Loading commonsense_qa train and valid datasets')

commonsense_qa_train_dataset = nlp.load_dataset('commonsense_qa', split = nlp.Split.TRAIN)
commonsense_qa_valid_dataset = nlp.load_dataset('commonsense_qa', split = nlp.Split.VALIDATION)

train_dataset = commonsense_qa_train_dataset.map(generate_concepts, batched = True, remove_columns = commonsense_qa_train_dataset.column_names)        
train_dataset = train_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

valid_dataset = commonsense_qa_valid_dataset.map(generate_concepts, batched = True, remove_columns = commonsense_qa_valid_dataset.column_names) 
valid_dataset = valid_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'attention_mask']
train_dataset.set_format(type='torch', columns = columns)
valid_dataset.set_format(type='torch', columns = columns)   
print('Processed {:d} training examples and {:d} validation examples'.format(len(train_dataset), len(valid_dataset)))

data_dir = os.path.join('../data', 'commonsense_qa_concepts')
if not os.path.exists(data_dir): os.makedirs(data_dir)
print('Saving train and validation files to {}'.format(data_dir))
torch.save(train_dataset, os.path.join(data_dir, TRAIN_FILE))
torch.save(valid_dataset, os.path.join(data_dir, VAL_FILE))
