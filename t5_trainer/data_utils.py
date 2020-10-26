import nlp
import torch

import os

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')

TRAIN_FILE = 'train_data.pt'
VAL_FILE = 'valid_data.pt'    

# process the examples in input and target text format. eos token added by t5 tokenizer.
def format_example(example):
    options = ['%s: %s' % (i, option) for i, option in zip(example['choices']['label'], example['choices']['text'])]
    example['input_text'] = 'question: %s  options: %s' % (example['question'], " ".join(options))
    example['target_text'] = '%s' % example['answerKey']
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], truncation = True,
                                                          pad_to_max_length = True, max_length = 256)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], truncation = True,
                                                          pad_to_max_length = True, max_length = 16)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

print('Getting data from nlp datasets')
train_dataset = nlp.load_dataset("commonsense_qa", split = nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset("commonsense_qa", split = nlp.Split.VALIDATION)
        
train_dataset = train_dataset.map(format_example)
train_dataset = train_dataset.map(convert_to_features, batched = True)

valid_dataset = valid_dataset.map(format_example, load_from_cache_file = False)
valid_dataset = valid_dataset.map(convert_to_features, batched = True, load_from_cache_file = False)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
train_dataset.set_format(type='torch', columns = columns)
valid_dataset.set_format(type='torch', columns = columns)   
print('Processed {:d} training examples and {:d} validation examples'.format(len(train_dataset), len(valid_dataset)))

print('Saving train and validation files to {}'.format("./data/csqa"))
if not os.path.exists('./data/csqa'): os.makedirs('./data/csqa')
torch.save(train_dataset, os.path.join("./data/csqa", TRAIN_FILE))
torch.save(valid_dataset, os.path.join("./data/csqa", VAL_FILE))