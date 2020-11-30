## Fine-tuning results from test runs

## commonsense_qa dataset

#### Effect of different types of target format on accuracy

Note: T5-base model fine-tuned for 3 epochs.

`question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? options: A: bank B: library C: department store D: mall E: new york`

| Target Format  | Accuracy     |
| ------------- |:-------------:|
| A: bank       | 62.40         |
| A             | 61.83         |
| bank          | 60.60         |   

#### Hyperparameter Tuning

Note: T5-base model fine-tuned for 3 epochs for each run using the target format `A: bank`.

| Batch size    | Learning rate | Accuracy      |
| ------------- |:-------------:|:-------------:|
| 8             | 1e-4          | 62.40         |
| 16            | 1e-4          | 61.26         |
| 8             | 5e-5          | 59.37         |
| 4             | 1e-4          | 61.58         |

#### Fine-tuning beyond 3 epochs

Note: T5-base model fine-tuned for each run using the target format `A: bank` and using `batch_size=8` and `learning_rate=1e-4`. 

Accuracy| Task  | Checkpoint | TS |
|---|---| ---| --- | 
| 0.566749| commonsense_qa|300| Tue Nov 17 15:14:11 PST 2020 |
| 0.610156| commonsense_qa|600| Tue Nov 17 15:14:11 PST 2020 |
| 0.610156| commonsense_qa|900| Tue Nov 17 15:14:11 PST 2020 |
| 0.609337| commonsense_qa|1200| Tue Nov 17 15:14:11 PST 2020 |
| 0.628174| commonsense_qa|1500| Tue Nov 17 15:14:11 PST 2020 |
| 0.620803| commonsense_qa|1800| Tue Nov 17 15:14:11 PST 2020 |
| 0.631450| commonsense_qa|2100| Tue Nov 17 15:14:11 PST 2020 |
| 0.626536| commonsense_qa|2400| Tue Nov 17 15:14:11 PST 2020 |
| 0.624079| commonsense_qa|2700| Tue Nov 17 15:14:11 PST 2020 |
| 0.624898| commonsense_qa|3000| Tue Nov 17 15:14:11 PST 2020 |
| 0.625717| commonsense_qa|10 epochs| Tue Nov 17 15:14:11 PST 2020 |

Note: T5-base model fine-tuned using the target format `A: bank` and using `batch_size=8` and `learning_rate=1e-4`. Non-stop run from T5-base.

Accuracy| Task  | Checkpoint |
|---|---| ---|
| 0.6224| commonsense_qa|1000 |
| 0.6216| commonsense_qa|2000 |
| 0.6289| commonsense_qa|3000 |
| 0.6167| commonsense_qa|4000 |
| 0.6175| commonsense_qa|5000 |
| 0.6232| commonsense_qa|6000 |
| 0.6224| commonsense_qa|20 epochs |

## social_i_qa dataset

#### Effect of different types of target format on accuracy

Note: T5-base model fine-tuned for 2 epochs.

`question: What will Robin want to do next? context: Robin left food out for the animals in her backyard to come and enjoy. options: A: chase the animals away B: watch the animals eat C: go out in the backyard`

| Target Format             | Accuracy      |
| --------------------------|:-------------:|
| B: watch the animals eat  | 63.71         |
| B                         | 65.86         |


Note: T5-base model fine-tuned for 2 epochs for each run using the target format `B: watch the animals eat`.

|Epochs         | Batch size    | Learning rate | Accuracy      |
| ------------- | ------------- |:-------------:|:-------------:|
|2              | 8             | 1e-4          | 63.56         |
|2              | 16            | 1e-4          | 63.15         |
|2              | 8             | 5e-5          | 61.67         |
|2              | 8             | 2e-5          | 57.98         |
|2              | 4             | 1e-4          | 65.35         |

I tried a few differente pochs.

|Epochs        | Batch size    | Checkpoint    | Learning rate | Accuracy      |
| -------------| ------------- |:-------------:|:-------------:|:-------------:|
|2             | 4             |(Final)        | 1e-4          | 65.35         |
|3             | 4             |(Final)        | 1e-4          | 65.51         |
|4             | 4             |(Final)        | 1e-4          | 64.53         |
|5             | 4             |(Final)        | 1e-4          | 63.61         |
|4 (non-stop)  | 4             |(Final)        | 2e-5          | 63.15         |
|4 (non-stop)  | 8             |1,200          | 1e-4          | 63.25         |
|4 (non-stop)  | 8             |2,400          | 1e-4          | 66.63         |
|4 (non-stop)  | 8             |3,600          | 1e-4          | 66.22         |
|4 (non-stop)  | 8             |(Final)        | 1e-4          | 66.58         |
|10(non-stop)  | 8             |(Final)        | 1e-4          | 66.63         |

## Transfer Learning: Trained on social_i_qa, finetuned on commonsense_qa

Take the T5 model trained on social_i_qa. Fine-tune the model on commonsense_qa and record accuracies.

| siqa epochs  | siqa Batch size| siqa learning rate | siqa Accuracy | csqa finetune epochs | csqa Accuracy |
| -------------| -------------- |:------------------:|:-------------:| :-------------------:|:-------------:|
|2             | 8              | 1e-4               | 63.56         | 0                    | 38.41         |
|2             | 8              | 1e-4               | 63.56         | 1                    | 59.87         |
|2             | 8              | 1e-4               | 63.56         | 2 (non-stop)         | 61.51         |
|2             | 8              | 1e-4               | 63.56         | 3 (non-stop)         | 61.18         |
|2             | 4              | 1e-4               | 65.35         | 0                    | 37.18         |
|3             | 4              | 1e-4               | 65.51         | 0                    | 38.00         |
|4             | 4              | 1e-4               | 64.53         | 0                    | 39.89         |
|5             | 4              | 1e-4               | 63.61         | 0                    | 38.98         |
|4 (non-stop)  | 4              | 2e-5               | 63.15         | 0                    | 38.82         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 0                    | 40.13         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 1                    | 58.55         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | (100 steps)          | 54.14         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | (300 steps)          | 59.54         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | (500 steps)          | 61.18         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 2                    | 60.69         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 3 (non-stop)         | 61.59         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 10(non-stop)         | 61.75         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 10(3 then 7)         | 62.82         |
|10(non-stop)  | 8              | 1e-4               | 66.63         | 3                    | 60.44         |
|10(non-stop)  | 8              | 1e-4               | 66.63         | (912 steps)          | 60.11         |
|10(non-stop)  | 8              | 1e-4               | 66.63         | (1,824 steps)        | 60.77         |
|10(non-stop)  | 8              | 1e-4               | 66.63         | (2,736 steps)        | 61.10         |
|10(non-stop)  | 8              | 1e-4               | 66.63         | 10(non-stop)         | 61.50         |

### Warmup Steps

Working on a model trained on social iqa (4 epochs non-stop, batch 8, lr 1e-4)

| Warmup Steps  | csqa finetune epochs | csqa Accuracy |
|:-------------:| :-------------------:|:-------------:|
| 100           | 1                    | 59.71         |
| 200           | 3 (out of 3 nonstop) | 62.00         |
| 1,500         | 10                   | 61.75         |


### 10 epochs checkpoints 

Commonsense finetuned (10 epochs, batch size 8, lr 1e-4, wu 0) on top of model trained on Social IQA (4 epochs, batch 8, lr 1e4)

| Checkpoint | Accuracy  |
|---|---|
| 300| 0.612613| 
| 600| 0.615070| 
| 900| 0.612613| 
| 1200| 0.609337| 
| 1500| 0.619984| 
| 1800| 0.624079| 
| 2100| 0.622441| 
| 2400| 0.615070| 
| 2700| 0.622441| 
| 3000| 0.622441| 



## cosmos_qa dataset


|Epochs         | Target Format             | Batch size    | Learning rate | Accuracy      |
| ------------- | --------------------------| ------------- |:-------------:|:-------------:|
|3              | B: watch the animals eat  | 8             | 1e-4          | 66.53         |

### transfer learning

Batch size 8

|Cosmos Epochs| Batch size| LR   | Cosmos Accuracy | Commonsense Epochs | Commonsense LR | Commonsense Accuracy | 
| ------------| --------- |:----:|:---------------:| :-----------------:| :-------------:| :-------------------:|
|3            | 8         | 1e-4 | 66.53           | 10                 | 1e-4           |  61.43               |
|3            | 8         | 1e-4 | 66.53           | 10                 | 5e-5           |  62.82               |

Since we achieved a higher accuracy than baseline for commonsense-over-Cosmos, we looked at how this accuracy changed over different checkpoints. Below is the table for finetuning Commonsense QA with LR = 5e-5, batch size 8, for 10 epochs.

Accuracy| Task  | Dir | Checkpoint | TS |
|---|---| ---| --- | ---|
| 0.552007| commonsense_qa| cs_on_cosmos|300| Sun Nov 15 23:13:18 PST 2020 |
| 0.597052| commonsense_qa| cs_on_cosmos|600| Sun Nov 15 23:13:18 PST 2020 |
| 0.609337| commonsense_qa| cs_on_cosmos|900| Sun Nov 15 23:13:18 PST 2020 |
| 0.615889| commonsense_qa| cs_on_cosmos|1200| Sun Nov 15 23:13:18 PST 2020 |
| 0.624079| commonsense_qa| cs_on_cosmos|1500| Sun Nov 15 23:13:18 PST 2020 |
| 0.628174| commonsense_qa| cs_on_cosmos|1800| Sun Nov 15 23:13:18 PST 2020 |
| 0.624898| commonsense_qa| cs_on_cosmos|2100| Sun Nov 15 23:13:18 PST 2020 |
| 0.628993| commonsense_qa| cs_on_cosmos|2400| Sun Nov 15 23:13:18 PST 2020 |
| 0.630631| commonsense_qa| cs_on_cosmos|2700| Sun Nov 15 23:13:18 PST 2020 |
| 0.627355| commonsense_qa| cs_on_cosmos|3000| Sun Nov 15 23:13:18 PST 2020 |
| 0.628174| commonsense_qa| cs_on_cosmos|End of 10 epochs| Sun Nov 15 23:13:18 PST 2020 |



## hellaswag dataset


|Epochs         | Batch size    | Learning rate | Accuracy      |
| ------------- | ------------- |:-------------:|:-------------:|
|3              | 4             | 1e-4          | 46.7835       |

### transfer learning

Batch size=8, learning rate=1e-4, warmup_steps=0

Accuracy| Task  | Checkpoint | TS |
|---|---| ---| --- | 
| 0.577396| commonsense_qa|cs_on_hellaswag/300| Wed Nov 25 22:53:08 PST 2020 |
| 0.612613| commonsense_qa|cs_on_hellaswag/600| Wed Nov 25 22:53:08 PST 2020 |
| 0.604423| commonsense_qa|cs_on_hellaswag/900| Wed Nov 25 22:53:08 PST 2020 |
| 0.619984| commonsense_qa|cs_on_hellaswag/1200| Wed Nov 25 22:53:08 PST 2020 |
| 0.606061| commonsense_qa|cs_on_hellaswag/1500| Wed Nov 25 22:53:08 PST 2020 |
| 0.617527| commonsense_qa|cs_on_hellaswag/1800| Wed Nov 25 22:53:08 PST 2020 |
| 0.610975| commonsense_qa|cs_on_hellaswag/2100| Wed Nov 25 22:53:08 PST 2020 |
| 0.620803| commonsense_qa|cs_on_hellaswag/2400| Wed Nov 25 22:53:08 PST 2020 |
| 0.619165| commonsense_qa|cs_on_hellaswag/2700| Wed Nov 25 22:53:08 PST 2020 |
| 0.626536| commonsense_qa|cs_on_hellaswag/3000| Wed Nov 25 22:53:08 PST 2020 |

Batch size=8, learning rate=1e-4, warmup_steps=150

Accuracy| Task  | Checkpoint | TS |
|---|---| ---| --- | 
| 0.564292| commonsense_qa|cs_on_hellaswag/checkpoint-300| Sun Nov 29 18:15:51 PST 2020 |
| 0.628174| commonsense_qa|cs_on_hellaswag/checkpoint-600| Sun Nov 29 18:15:51 PST 2020 |
| 0.615070| commonsense_qa|cs_on_hellaswag/checkpoint-900| Sun Nov 29 18:15:51 PST 2020 |
| 0.627355| commonsense_qa|cs_on_hellaswag/checkpoint-1200| Sun Nov 29 18:15:51 PST 2020 |
| 0.624898| commonsense_qa|cs_on_hellaswag/checkpoint-1500| Sun Nov 29 18:15:51 PST 2020 |
| 0.621622| commonsense_qa|cs_on_hellaswag/checkpoint-1800| Sun Nov 29 18:15:51 PST 2020 |
| 0.624898| commonsense_qa|cs_on_hellaswag/checkpoint-2100| Sun Nov 29 18:15:51 PST 2020 |
| 0.623260| commonsense_qa|cs_on_hellaswag/checkpoint-2400| Sun Nov 29 18:15:51 PST 2020 |
| 0.626536| commonsense_qa|cs_on_hellaswag/checkpoint-2700| Sun Nov 29 18:15:51 PST 2020 |
| 0.628993| commonsense_qa|cs_on_hellaswag/checkpoint-3000| Sun Nov 29 18:15:51 PST 2020 |
| 0.630631| commonsense_qa|End of 10 epochs | Sun Nov 29 18:15:51 PST 2020 |

Batch size=8, learning rate=1e-4, warmup_steps=200

Accuracy| Task  | Checkpoint | TS |
|---|---| ---| --- | 
| 0.561835| commonsense_qa|cs_on_hellaswag/checkpoint-300| Mon Nov 30 13:14:15 PST 2020 |
| 0.612613| commonsense_qa|cs_on_hellaswag/checkpoint-600| Mon Nov 30 13:14:15 PST 2020 |
| 0.611794| commonsense_qa|cs_on_hellaswag/checkpoint-900| Mon Nov 30 13:14:15 PST 2020 |
| 0.628174| commonsense_qa|cs_on_hellaswag/checkpoint-1200| Mon Nov 30 13:14:15 PST 2020 |
| 0.629812| commonsense_qa|cs_on_hellaswag/checkpoint-1500| Mon Nov 30 13:14:15 PST 2020 |
| 0.637183| commonsense_qa|cs_on_hellaswag/checkpoint-1800| Mon Nov 30 13:14:15 PST 2020 |
| 0.623260| commonsense_qa|cs_on_hellaswag/checkpoint-2100| Mon Nov 30 13:14:15 PST 2020 |
| 0.636364| commonsense_qa|cs_on_hellaswag/checkpoint-2400| Mon Nov 30 13:14:15 PST 2020 |
| 0.634726| commonsense_qa|cs_on_hellaswag/checkpoint-2700| Mon Nov 30 13:14:15 PST 2020 |
| 0.640459| commonsense_qa|cs_on_hellaswag/checkpoint-3000| Mon Nov 30 13:14:15 PST 2020 |
| 0.640459| commonsense_qa|End of 10 epochs| Mon Nov 30 13:14:15 PST 2020 |

## Sample Efficiency

#### Effect of different fractions of training data on Accuracy

Note: T5-base model fine-tuned for 10 epochs, using the target format `A: bank` and using `batch_size=8` and `learning_rate=1e-4`.

| Model                | 20%    | 40%    | 60%    | 80%    | 100%   | Epochs | LR     | Run |
| -------------------  |:------:|:------:|:------:|:------:|:------:| :-----:| :-----:| :--: |
| T5-base              | 56.67  | 59.95  | 61.26  | 62.40  | 62.65  | 10     | 1e-4   | Sonali |
| T5-base + cosmos_qa  | 56.67  | 59.05  | 61.10  | 62.08  | 62.82  | 10     | 5e-5   | Haerang|
| T5-base + cosmos_qa  | 58.39  | 59.70  | 61.26  | 61.91  | 62.82  | 10     | 1e-4   | Sonali|
| T5-base + social_i_qa| 57.74  | 59.46  | 60.85  | 60.77  | 62.00  |  10    | 1e-4   | Haerang|
| T5-base + social_i_qa| 58.55  | 60.44  | 60.11  | 62.16  | 62.00  |  10    | 1e-4   | Sonali|
| T5-base + hellaswag  | 56.42  | 58.06  | 60.85  | 62.24  | 62.65  |  10    | 1e-4   | Sonali|
| T5-base + hellaswag  | 56.51  | 58.64  | -  | -  | -  |  10    | 1e-4   | ws=150|
