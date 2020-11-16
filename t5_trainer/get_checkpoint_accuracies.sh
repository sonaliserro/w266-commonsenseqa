
#!/bin/bash

task="commonsense_qa"
model_type_directory="cs_on_cosmos"
model_output_directory="batch_8_lr_5e-5_wu_0_epochs10"
max_target_length=55

# setup variables 
ts=$(TZ=":US/Pacific" date )
output_dir="./models/${model_type_directory}/${model_output_directory}"
filename=checkpoint_accuracies
FILE=${output_dir}/${filename}.md

echo "Task: ${task}"
echo "Output directory: ${model_output_directory}"

# Create the results md file or append results at the bottom 
if [ -f "$FILE" ]; then
    echo "${FILE} exists."
    echo "|---|---| ---| --- | ---|"  >> ${FILE}
else 
    touch ${FILE}
    echo "Created ${FILE}"
    echo "Accuracy| Task  | Checkpoint | TS |" >> ${FILE}
    echo "|---|---| ---| --- | ---|"  >> ${FILE}
fi


for checkpoint_dir in ${output_dir}/checkpoint*/; do
    echo "$checkpoint_dir"
   
    # Evaluate and save accuracy
    echo "{\"model_name_or_path\": \"${checkpoint_dir}\",
            \"valid_file_path\": \"./data/${task}/valid_data.pt\",
            \"tokenizer_name_or_path\": \"${checkpoint_dir}\",
            \"max_target_length\": ${max_target_length}}" > eval_args.json
    python ./t5_eval.py 

    accuracy=$(head -1 ${checkpoint_dir}/eval_accuracy.txt | cut -d ' ' -f3)
    echo "accuracy: ${accuracy}"
    echo "| ${accuracy}| ${task}|${checkpoint_dir}| ${ts} |" >> ${FILE}
    
done


