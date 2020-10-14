### Install packages
pip install -r requirements.txt

### Configure wandb
python
>>> import wandb
>>> wandb.login()

### Download the commomsense_qa dataset from transformers nlp
python ./data_utils.py

### Train the t5 model using the configuration specified in args.json
### Models and checkpoints saved under ./models
python ./t5_train.py

### Bring up Tensorboard    
tensorboard --logdir ./runs/

### Evaluation notebook t5_eval.ipynb
