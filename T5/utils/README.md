## Generate leaderboard predictions

1. Edit the generate_leaderboard_predictions.py to reference the appropriate model directory.

2. Run the script
```
python generate_leadeboard_predictions.py
```

3. It should generate a `predictions_leaderboard.csv` file in the current directory.

The following are results from the leaderboard evaluation,

### Baseline

Results for T5-Base Model. Model available on Google Storage at `gs://w266-commonsenseqa/models/T5/commonsense_qa/10_epochs`.

```
{"accuracy": 0.5596491228070175}
```

### Best

Results for T5-FT-HellaSwag. Model available on Google Storage at `gs://w266-commonsenseqa/models/T5/cs_on_hellaswag/10_epochs`.

```
 {"accuracy": 0.5535087719298246}
```
