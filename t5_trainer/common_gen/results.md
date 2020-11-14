## Using common_gen to generate choice-specific context for the commonsense_qa task

#### Baseline

Results of T5-base fine-tuned with commonsense_qa dataset using the target format `A`.

Input format: `question: John loved to paint houses. How did he usually do it? options: A: clothes get stained B: with brush C: wallpaper D: electrical circuit E: draw`

Hyperparametes: `batch-size=8, learning_rate=e1-4`

| Epochs        | Accuracy      |
| ------------- |:-------------:|
| 3 epochs      | 61.83         |
| 10 epochs     | 62.73         |


#### Using common_gen choice-specific context

Input format: `question: John loved to paint houses. How did he usually do it? options: A: clothes get stained | i loved getting my clothes stained and painted for my house B: with brush | i loved painting houses with a brush C: wallpaper | i loved the idea of painting wallpaper on a house D: electrical circuit | john loved painting electrical circuits in his houses E: draw | i loved the drawing and painting of the houses`

Hyperparametes: `batch-size=8, learning_rate=e1-4`

| Epochs        | Accuracy      |
| ------------- |:-------------:|
| 3 epochs      | 62.48         |
| 10 epochs     | 62.48         |

We also observe that the model converges faster using the common_gen choice-specific context