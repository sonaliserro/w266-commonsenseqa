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

#### Fine-tuning beyond 3 epochs

Note: T5-base model fine-tuned for each run using the target format `A: bank` and using `batch_size=8` and `learning_rate=1e-4`..

| Epochs        | Steps         | Accuracy      |
| ------------- |:-------------:|:-------------:|
| 3             | 912           | 62.40         |
| -             | 912+500       | 62.73         |
| -             | 912+1000      | 62.89         |
| -             | 912+1500      | 63.06         |
| -             | 912+2000      | 64.29         |
| 10            | 2128          | 64.12         |
| -             | 2128+500      | 61.83         |
| 13            | 2128+912      | 62.24         |

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

Keeping the batch size at 4 and learning rate at 1e-4, I tried a few differente pochs.

|Epochs        | Batch size    | Learning rate | Accuracy      |
| -------------| ------------- |:-------------:|:-------------:|
|2             | 4             | 1e-4          | 65.35         |
|3             | 4             | 1e-4          | 65.51         |
|4             | 4             | 1e-4          | 64.53         |
|5             | 4             | 1e-4          | 63.61         |
|4 (non-stop)  | 4             | 2e-5          | 63.15         |
|4 (non-stop)  | 8             | 1e-4          | 66.58         |

## Transfer Learning: Trained on social_i_qa, finetuned on commonsense_qa

Take the T5 model trained on social_i_qa. Fine-tune the model on commonsense_qa and record accuracies.

| siqa epochs  | siqa Batch size| siqa learning rate | siqa Accuracy | csqa finetune epochs | csqa Accuracy |
| -------------| -------------- |:------------------:|:-------------:| :-------------------:|:-------------:|
|2             | 8              | 1e-4               | 63.56         | 0                    | 38.41         |
|2             | 8              | 1e-4               | 63.56         | 1                    | 59.87         |
|2             | 8              | 1e-4               | 63.56         | 2 (non-stop)         | 61.51         |
|2             | 8              | 1e-4               | 63.56         | 3 (non-stop)         | 61.18         |
|2             | 4              | 1e-4               | 65.35         | 0                    | 37.18         |
|2             | 4              | 1e-4               | 65.35         | 1                    |               |
|2             | 4              | 1e-4               | 65.35         | 2                    |               |
|2             | 4              | 1e-4               | 65.35         | 3                    |               |
|3             | 4              | 1e-4               | 65.51         | 0                    | 38.00         |
|4             | 4              | 1e-4               | 64.53         | 0                    | 39.89         |
|5             | 4              | 1e-4               | 63.61         | 0                    | 38.98         |
|4 (non-stop)  | 4              | 2e-5               | 63.15         | 0                    | 38.82         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 0                    | 40.13         |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 1                    |               |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 2 (non-stop)         |               |
|4 (non-stop)  | 8              | 1e-4               | 66.58         | 3 (non-stop)         |               |


#### Effect of different types of target format on accuracy

## Sample Efficiency

#### Effect of different fractions of training data on Accuracy

Note: T5-base model fine-tuned for 10 epochs, using the target format `A: bank` and using `batch_size=8` and `learning_rate=1e-4`.

| Model         | 20%    | 40%    | 60%    | 80%    | 100%   |
| --------------|:------:|:------:|:------:|:------:|:------:|
| T5-base       | 56.67  | 59.95  | 61.26  | 62.40  | 64.12  |
