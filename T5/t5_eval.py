'''
Utility to evaluate a T5 model using the provided dataset. 

By passing in "--do_eval", the model will evaluate the accuracy of the 
provided validation dataset. The predictions are compared to the targets 
and the model accuracy is saved in a file "eval_accuracy.txt" file in the 
same directory as the model.

By passing in "--do_predict", the model will make predictions for the
provided test dataset. The predictions are saved in a file "predictions.txt" 
file in the same directory as the model. 
'''
from dataclasses import dataclass, field
from typing import Optional

import torch

import os
import sys

from tqdm.auto import tqdm

import transformers

from transformers import (
    HfArgumentParser,
    DataCollator,
    T5ForConditionalGeneration, 
    T5Tokenizer
)

from sklearn import metrics

# Make the logging level as INFO
transformers.logging.set_verbosity_info()

@dataclass
class EvalArguments:
    
    model_name_or_path: str = field(
        metadata = {'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    file_path: str = field(
        metadata = {'help': 'Path for dataset to use'}
    )    
    tokenizer_name_or_path: Optional[str] = field(
        default = None, metadata = {'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    max_target_length: Optional[int] = field(
        default = 16, metadata = {'help': 'maximum length for decoding'}
    )
    do_eval: Optional[bool] = field(
        default = False, metadata={'help': 'Whether to run eval on the dev set.'}
    )
    do_predict: Optional[bool] = field(
        default = False, metadata={'help': 'Whether to run predictions on the test set.'}
    )
    early_stopping: Optional[bool] = field(
        default = False, metadata={'help': 'Finish generation when all beam hypotheses reached the EOS token.'}
    )
    num_beams: Optional[int] = field(
        default = 1, metadata={'help': 'Number of beams for beam search. 1 means no beam search.'}
    )
    no_repeat_ngram_size: Optional[int] = field(
        default = 0, metadata={'help': 'If set to int > 0, all ngrams of that size can only occur once.'}
    )
    length_penalty: Optional[int] = field(
        default = 1.0, metadata={'help': 'Set to values < 1.0 in order to encourage the model to generate shorter sequences.'}
    )
        
def main(args):
    parser = HfArgumentParser((EvalArguments,))
    
    # Read command-line arguments if present, else read arguments from json file
    if len(args) >= 2:
        args = parser.parse_args_into_dataclasses(args = args)[0]
    else:
        args = parser.parse_json_file(json_file = os.path.abspath('eval_args.json'))[0]
    
    # Initiliaze the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(                
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    )
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Load the dataset
    dataset = torch.load(args.file_path)          
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16)

    # Evaluate accuracy on the dev set
    if args.do_eval:
        predictions = []
        targets = []
        model.to(device)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                prediction = model.generate(
                    input_ids = batch['input_ids'].to(device), 
                    attention_mask = batch['attention_mask'].to(device),
                    max_length = args.max_target_length
                )

                prediction = [tokenizer.decode(ids) for ids in prediction]
                target = [tokenizer.decode(ids) for ids in batch['target_ids']]

                predictions.extend(prediction)
                targets.extend(target)

        accuracy = metrics.accuracy_score(targets, predictions)
        output_file = os.path.join(args.model_name_or_path, 'eval_accuracy.txt')
        with open(output_file, 'w') as writer:
            writer.write('Accuracy = %f\n' % accuracy)
      
    # Generate predictions for provided file
    if args.do_predict:
        predictions = []
        model.to(device)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                prediction = model.generate(
                    input_ids = batch['input_ids'].to(device), 
                    attention_mask = batch['attention_mask'].to(device),
                    max_length = args.max_target_length,
                    num_beams = args.num_beams,
                    early_stopping = args.early_stopping,
                    no_repeat_ngram_size = args.no_repeat_ngram_size,
                    length_penalty = args.length_penalty
                )

                prediction = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in prediction]
                predictions.extend(prediction)
                
        output_file = os.path.join(args.model_name_or_path, 'predictions.txt')
        with open(output_file, 'w') as writer:
            writer.write('\n'.join(predictions))

if __name__ == '__main__':
    main(sys.argv[1:])
