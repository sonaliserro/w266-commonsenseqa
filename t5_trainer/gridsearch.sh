
#!/bin/bash

ts=$(date +%s)
filename=gridsearch_${ts}
touch ${filename}.md

task="cosmos_qa"
model_output_directory="cosmos_qa"
echo "Task: ${task}" >> ${filename}.md
echo "Output directory: ${model_output_directory}" >> ${filename}.md

echo "|Epochs         | Batch size    | Learning rate | Accuracy      |" >> ${filename}.md


batchsizes=( 8 12)
for s in "${batchsizes[@]}"
do
    learningrates=( 1e-4 5e-5 2e-5 )

    for l in "${learningrates[@]}"
    do
        epochs=( 1 )

        for e in "${epochs[@]}"
        do
            echo $"{\"num_cores\": 8,
                    \"training_script\": \"t5_train.py\",
                    \"model_name_or_path\": \"t5-base\",
                    \"task_name\": \"${task}\",
                    \"output_dir\": \"./models/${model_output_directory}/batch_${s}_lr_${l}_epochs${e}\",
                    \"overwrite_output_dir\": false,
                    \"per_device_train_batch_size\": ${s},
                    \"per_device_eval_batch_size\": ${s},
                    \"gradient_accumulation_steps\": 4,
                    \"learning_rate\": ${l},
                    \"num_train_epochs\": ${e},
                    \"do_train\": true,
                    \"do_eval\": true,
                    \"evaluate_during_training\": true,
                    \"logging_steps\": 600,
                    \"save_steps\": 1500}" > training_args.json
            python ./t5_train.py 
            
            echo "{\"model_name_or_path\": \"./models/${model_output_directory}/batch_${s}_lr_${l}_epochs${e}\",
                    \"valid_file_path\": \"./data/${task}/valid_data.pt\",
                    \"tokenizer_name_or_path\": \"./models/${model_output_directory}/batch_${s}_lr_${l}_epochs${e}\",
                    \"max_target_length\": 55}" > eval_args.json
            python ./t5_eval.py 
            
            
             accuracy=$(head -1 ./models/${model_output_directory}/batch_${s}_lr_${l}_epochs${e}/eval_accuracy.txt | cut -d ' ' -f3)
            echo "accuracy: ${accuracy}"
            echo "|${e}              | ${s}             | ${l}         | ${accuracy}      |" >> ${filename}.md
            
        done
    done
done