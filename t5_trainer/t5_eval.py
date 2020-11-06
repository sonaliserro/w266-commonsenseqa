import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import os

from tqdm.auto import tqdm

import transformers

from transformers import (
    HfArgumentParser,
    DataCollator,
    T5ForConditionalGeneration, 
    T5Tokenizer
)

from sklearn import metrics

logger = logging.getLogger(__name__)

# Make the logging level as INFO
transformers.logging.set_verbosity_info()

@dataclass
class EvalArguments:
    
    model_name_or_path: str = field(
        metadata = {'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    valid_file_path: str = field(
        metadata = {'help': 'Path for cached valid dataset'}
    )    
    tokenizer_name_or_path: Optional[str] = field(
        default = None, metadata = {'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    max_target_length: Optional[int] = field(
        default = 16, metadata = {'help': 'maximum length for decoding'}
    )
    do_eval: Optional[bool] = field(
        default = True, metadata={'help': 'Whether to run eval on the dev set.'}
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
        
def main():
    parser = HfArgumentParser((EvalArguments,))
    args = parser.parse_json_file(json_file = os.path.abspath('eval_args.json'))[0]
    
    # Initiliaze the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(                
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    )
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Load the validation dataset
    valid_dataset = torch.load(args.valid_file_path)          
    dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 16)

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

                prediction = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in prediction]
                predictions.extend(prediction)
                
        output_file = os.path.join(args.model_name_or_path, 'predictions.txt')
        with open(output_file, 'w') as writer:
            writer.write('\n'.join(predictions))          
            

if __name__ == '__main__':
    main()
