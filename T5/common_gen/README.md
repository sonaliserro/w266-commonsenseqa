#### Steps to evaluate the common_gen predictions using the nlg-eval package

1. Install the following dependency
```
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```

2. Set up nlg-eval
```
nlg-eval --setup
```

3. Make sure you have a predictions.txt file that was generated using the `t5_eval.py` script

2. Run the following script, that will generate a `references.txt` file and print the results in an `nlg_eval.txt` file
```
python ./eval_common_gen.py
```

#### Steps to generate commonsense_qa concepts sets for common_gen

1. Install the spacy package
```
pip install -U spacy
```

2. Download the en_core_web_lg language model
```
python -m spacy download en_core_web_lg
```

3. Run the following script that will write tokenized pytorch training and validation file(s) that will be used as input to the model and saved at `../data/commonsense_qa_concepts`.
```
python ./generate_concepts.py
```

4. Generate predictions for the above files using the `t5_eval.py` script using the T5-base model that was fine-tuned on common_gen. This will generate a `predictions.txt` file in the model directory. Use the following arguments, 
```
{
	"model_name_or_path": "./models/common_gen/3_epochs",
	"valid_file_path": "./data/commonsense_qa_concepts/train_data.pt",
	"tokenizer_name_or_path": "./models/common_gen/3_epochs",
	"max_target_length": 32,
	"do_eval": false,
	"do_predict": true,
	"num_beams": 5,
	"early_stopping": true,
	"no_repeat_ngram_size": 3,
	"length_penalty": 0.6
}
```

5. Run the following script to generate commonsense_qa training and validation datasets that contain the generated sentences as hints along with each of the answer options. The files will get saved at `../data/commonsense_qa_concept_sentences`.
```
python ./generate_commonsense_qa_concept_sentences.py
```

6. Finally fine-tune a T5-base model with the dataset from above using the `t5_train.py` script. Use the following arguments, 
```
{
	"num_cores": 8,
	"training_script": "t5_train.py",
	"model_name_or_path": "t5-base",
	"task_name": "commonsense_qa_concept_sentences",
	"output_dir": "./models/commonsense_qa_concept_sentences",
	"overwrite_output_dir": true,
	"per_device_train_batch_size": 8,
	"per_device_eval_batch_size": 8,
	"gradient_accumulation_steps": 4,
	"learning_rate": 0.0001,
	"num_train_epochs": 3,
	"do_train": true,
	"do_eval": true,
	"evaluate_during_training": true,
	"logging_steps": 500,
	"save_steps": 1000
}
```