#### Steps to train T5 on the [commonsense_qa,social_i_qa] dataset (using the huggingface transformer model using pytorch)

1. Install packages

```pip install -r requirements.txt```

2. Configure wandb (You need to set up an account on https://wandb.ai/, and runs will be tracked under project-name `t5-hf-csqa`)

```
python
>>> import wandb
>>> wandb.login()
```

3. Download the [commomsense_qa, social_i_qa] dataset from the transformers nlp package. Formatted train and validation files torch files saved under ./data/[dataset_name]/. Supply relevant arguments in the `data_utils_args.json` file.

```python ./data_utils.py```

4. Train the t5 model using the configuration specified in `training_args.json` file. Models and checkpoints saved under ./models/[task_name]

```python ./t5_train.py```

5. Bring up Tensorboard

```tensorboard --logdir ./runs/```

6. Evalaute accuracy of the fine-tuned model using the configuration specified in `eval_args.json` file.

```python ./t5_eval.py```

7. Evaluation notebook t5_eval.ipynb

8. Fine-tuned models and processed dataset(s) are uploaded to Google Storage under `gs://w266-commonsenseqa/models/T5/`. To download the T5-base model fine-tuned on commonsense_qa for 3 epochs run the following command,

```gsutil cp gs://w266-commonsenseqa/models/T5/commonsense_qa/3_epochs/* .```
