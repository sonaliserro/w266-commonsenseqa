## Using common_gen to generate choice-specific context for the commonsense_qa task

#### Baseline

Results of T5-base fine-tuned with commonsense_qa dataset using the target format `A`.

Input format: `question: John loved to paint houses. How did he usually do it? options: A: clothes get stained B: with brush C: wallpaper D: electrical circuit E: draw`

Hyperparametes: `batch-size=8, learning_rate=e1-4`

| Epochs        | Accuracy      |
| ------------- |:-------------:|
| 3 epochs      | 61.83         |
| 10 epochs     | 62.73         |

Training for 10 epochs with checkpoints every 500 steps

Accuracy| Task  | Checkpoint | TS |
|---|---| ---| --- | 
| 0.597871| commonsense_qa|500 | Mon Nov 16 14:06:53 PST 2020 |
| 0.615889| commonsense_qa|1000| Mon Nov 16 14:06:53 PST 2020 |
| 0.629812| commonsense_qa|1500| Mon Nov 16 14:06:53 PST 2020 |
| 0.627355| commonsense_qa|2000| Mon Nov 16 14:06:53 PST 2020 |
| 0.622441| commonsense_qa|2500| Mon Nov 16 14:06:53 PST 2020 |
| 0.627355| commonsense_qa|3000| Mon Nov 16 14:06:53 PST 2020 |

#### Using common_gen choice-specific context

Input format: `question: John loved to paint houses. How did he usually do it? options: A: clothes get stained | i loved getting my clothes stained and painted for my house B: with brush | i loved painting houses with a brush C: wallpaper | i loved the idea of painting wallpaper on a house D: electrical circuit | john loved painting electrical circuits in his houses E: draw | i loved the drawing and painting of the houses`

Hyperparametes: `batch-size=8, learning_rate=e1-4`

| Epochs        | Accuracy      |
| ------------- |:-------------:|
| 3 epochs      | 62.48         |
| 10 epochs     | 62.48         |

Training for 10 epochs with checkpoints every 500 steps

Accuracy| Task  | Checkpoint | TS |
|---|---| ---| --- | 
| 0.597871| commonsense_qa_concept_sentences|500 | Mon Nov 16 13:46:40 PST 2020 |
| 0.624898| commonsense_qa_concept_sentences|1000| Mon Nov 16 13:46:40 PST 2020 |
| 0.610975| commonsense_qa_concept_sentences|1500| Mon Nov 16 13:46:40 PST 2020 |
| 0.624079| commonsense_qa_concept_sentences|2000| Mon Nov 16 13:46:40 PST 2020 |
| 0.619984| commonsense_qa_concept_sentences|2500| Mon Nov 16 13:46:40 PST 2020 |
| 0.624898| commonsense_qa_concept_sentences|3000| Mon Nov 16 13:46:40 PST 2020 |

We can observe that the model converges faster using the common_gen choice-specific context