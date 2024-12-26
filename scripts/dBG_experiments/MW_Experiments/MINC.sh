#!/bin/zsh

model_name=MICN

horizons=(24 48 96 168)

run_experiment() {
  local horizon=$1  # Accept horizon as a parameter
  experiment_tag="${model_name}_${horizon}"
  echo ">>>>>>>>>>>>> Now testing ${experiment_tag}"
  python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/MW \
  --seasonal_patterns MW \
  --model_id MW_${model_name} \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --tag $experiment_tag\
  --pred_len $horizon \
  --devices 3 \
  --loss 'SMAPE' \
  --train_epochs 30 \
  --patience 10
}

for horizon in "${horizons[@]}"; do
  run_experiment "$horizon"  # Pass horizon to the function
done
