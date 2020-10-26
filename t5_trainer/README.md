#### Steps to train T5 on the commonsense_qa dataset (using the huggingface transformer model using pytorch)

1. Install packages

```pip install -r requirements.txt```

2. Configure wandb (You need to set up an account on https://wandb.ai/, and runs will be tracked under project-name `t5-hf-csqa`)
```
python
>>> import wandb
>>> wandb.login()
```
3. Download the commomsense_qa dataset from the transformers nlp package. Formatted train and validation files torch files saved under ./data/csqa/

```python ./data_utils.py```

4. Train the t5 model using the configuration specified in args.json. Models and checkpoints saved under ./models

```python ./t5_train.py```

5. Bring up Tensorboard

```tensorboard --logdir ./runs/```

6. Evaluation notebook t5_eval.ipynb
