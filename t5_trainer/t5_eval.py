import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import os

from tqdm.auto import tqdm

from transformers import (
    HfArgumentParser,
    DataCollator,
    T5ForConditionalGeneration, 
    T5Tokenizer
)

from sklearn import metrics

logger = logging.getLogger(__name__)

@dataclass
class EvalArguments:
    
    model_name_or_path: str = field(
        metadata = {"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    valid_file_path: str = field(
        metadata = {"help": "Path for cached valid dataset"}
    )    
    tokenizer_name_or_path: Optional[str] = field(
        default = None, metadata = {"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_target_length: Optional[int] = field(
        default = 16, metadata = {"help": "maximum length for decoding"}
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

    # Generate predictions
    predictions = []
    targets = []
    model.to('cpu')
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            prediction = model.generate(
                input_ids = batch['input_ids'].to('cpu'), 
                attention_mask = batch['attention_mask'].to('cpu'),
                max_length = args.max_target_length
            )

            prediction = [tokenizer.decode(ids) for ids in prediction]
            target = [tokenizer.decode(ids) for ids in batch['target_ids']]
            
            predictions.extend(prediction)
            targets.extend(target)

    print("***** Accuracy *****")
    print(metrics.accuracy_score(targets, predictions))

if __name__ == "__main__":
    main()