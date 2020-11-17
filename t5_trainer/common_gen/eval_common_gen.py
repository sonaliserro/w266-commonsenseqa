import nlp

from nlgeval import compute_metrics

# Load the common_gen validation dataset
valid_dataset = nlp.load_dataset('common_gen', split = nlp.Split.VALIDATION)

# Generate a references.txt file that contains 1 reference per line
output_file = 'references.txt'
with open(output_file, 'w') as writer:
    for idx, row in enumerate(valid_dataset):
        writer.write(row['target'] + '\n')
        
# Call the compute_metrics method
metrics_dict = compute_metrics(hypothesis = '../models/common_gen/3_epochs/predictions.txt',
                               references = ['references.txt'],
                               no_skipthoughts = True,
                               no_glove = True)

# Print results
eval_file = 'nlg_eval.txt'
with open(eval_file, 'w') as writer:
    for key, value in metrics_dict.items():
        writer.write(key + ':' + str(value) + '\n')
