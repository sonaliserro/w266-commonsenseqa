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

3. Run the script that will write a tokenizer pytorch file to ../data/commonsense_qa_concepts
```
python ./generate_concepts.py
```

