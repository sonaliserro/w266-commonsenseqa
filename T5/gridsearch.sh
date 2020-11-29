
#!/bin/bash

task="commonsense_qa"
model_output_directory="cs_on_cosmos"

# setup variables 
ts=$(TZ=":US/Pacific" date )
filename=gridsearch_${model_output_directory}
FILE=./models/${model_output_directory}/${filename}.md

echo "Task: ${task}"
echo "Output directory: ${model_output_directory}"

# Create the results md file or append results at the bottom 
if [ -f "$FILE" ]; then
    echo "${FILE} exists."
    echo "|---|---| ---| --- | ---| ---  | --- | --- |"  >> ${FILE}
else 
    touch ${FILE}
    echo "Created ${FILE}"
    echo "|Epochs| Batch size| Warmup Steps| Learning rate | Accuracy| Task  | Dir | TS |" >> ${FILE}
    echo "|---|---| ---| --- | ---| ---  | --- | --- |"  >> ${FILE}
fi


# Loop through hyperparams 
batchsizes=( 8)

for s in "${batchsizes[@]}"
do 
    warmup_steps=( 0 200)
    for wu in "${warmup_steps[@]}"
    do
        learningrates=( 1e-4 5e-5 2e-5 )
        for l in "${learningrates[@]}"
        do
            epochs=(10)
            for e in "${epochs[@]}"
            do
                output_dir="./models/${model_output_directory}/batch_${s}_lr_${l}_wu_${wu}_epochs${e}"

                # train
                echo $"{\"num_cores\": 8,
                        \"training_script\": \"t5_train.py\",
                        \"model_name_or_path\": \"t5-base\",
                        \"task_name\": \"${task}\",
                        \"output_dir\": \"${output_dir}\",
                        \"overwrite_output_dir\": false,
                        \"per_device_train_batch_size\": ${s},
                        \"per_device_eval_batch_size\": ${s},
                        \"gradient_accumulation_steps\": 4,
                        \"learning_rate\": ${l},
                        \"num_train_epochs\": ${e},
                        \"num_warmup_steps\": ${wu},
                        \"do_train\": true,
                        \"do_eval\": true,
                        \"evaluate_during_training\": true,
                        \"logging_steps\": 150,
                        \"save_steps\": 300}" > training_args.json
                python ./t5_train.py 

                # Evaluate and save accuracy
                echo "{\"model_name_or_path\": \"${output_dir}\",
                        \"valid_file_path\": \"./data/${task}/valid_data.pt\",
                        \"tokenizer_name_or_path\": \"${output_dir}\",
                        \"max_target_length\": 55}" > eval_args.json
                python ./t5_eval.py 

                accuracy=$(head -1 ${output_dir}/eval_accuracy.txt | cut -d ' ' -f3)
                echo "accuracy: ${accuracy}"
                echo "|${e}     | ${s}         | ${wu}         | ${l}         | ${accuracy}| ${task}| ${model_output_directory}| ${ts} |" >> ${FILE}

            done
        done
    done
done
