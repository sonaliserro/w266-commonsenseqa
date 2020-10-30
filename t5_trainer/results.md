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
| 8             | 5e-5          | 59.00         |

#### Fine-tuning beyond 3 epochs

Note: T5-base model fine-tuned for each run using the target format `A: bank` and using `batch_size=8` and `learning_rate=1e-4`..

| Epochs        | Steps 	| Accuracy      |
| ------------- |:-------------:|:-------------:|
| 3             | 912           | 62.40         |
| -             | 912+500       | 62.73         |
| -             | 912+100       | 62.89         |
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

