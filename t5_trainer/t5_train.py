# Adapted from https://github.com/huggingface/transformers/blob/master/examples/
import logging
import os
import sys

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

import wandb

import transformers

from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    EvalPrediction,
    set_seed,
)

logger = logging.getLogger(__name__)

TRAIN_FILE = 'train_data.pt'
VAL_FILE = 'valid_data.pt'

TOKENIZER_PAD_TOKEN_ID = 0

# Make the logging level as INFO
transformers.logging.set_verbosity_info()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata = {"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default = None, metadata = {"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default = None, metadata = {"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
        
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """   
    task_name: str = field(
        metadata={"help": "Task name"},
    )    
    data_dir: Optional[str] = field(
        default = './data',
        metadata={"help": "Path for train and eval dataset(s)"}
    )

@dataclass
class T2TDataCollator:       
    # prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
    # this is necessacry because the trainer directly passes this dict as arguments to the model.
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        labels = torch.stack([example['target_ids'] for example in batch])
        # Do not calculate loss for pad tokens. All labels set to -100 are ignored (masked). 
        labels[labels[:, :] == TOKENIZER_PAD_TOKEN_ID] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': labels, 
            'decoder_attention_mask': decoder_attention_mask
        }

def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # Load the arguments from a json file
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('training_args.json'))

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
        
    # Log both gradients and parameters in weights and biases
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_PROJECT'] = '_'.join(['t5_train', data_args.task_name])

    # Load pretrained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
    )

    # Load the train/eval dataset(s)
    train_dataset = torch.load(os.path.join(data_args.data_dir, data_args.task_name, TRAIN_FILE)) if training_args.do_train else None
    valid_dataset = torch.load(os.path.join(data_args.data_dir, data_args.task_name, VAL_FILE)) if training_args.do_eval else None
    logger.info("Finished loading dataset(s)")

    # Initialize the Trainer
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        data_collator = T2TDataCollator(),
        prediction_loss_only = True
    )
    
    # Training
    if training_args.do_train:
        trainer.train(
            model_path = model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # Also re-save the tokenizer to the same directory.
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results

if __name__ == "__main__":
    main()
