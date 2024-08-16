#!/bin/bash

model_name=TimesNet
device_id=0
approximate=True
architectures=("append_linear")
k_params=(3 4)
disc_params=(15 20 25)
include_corr_params=("True" "False")
motif_count_params=(10 15 20)
graph_dim_params=(16 32 64 128)

run_experiment() {
      experiment_tag="A_${arch}_k${k_param}_d${disc}_ap${approximate}_gdim${dbgdim}_fdim${feat_count}_corr${include_corr}"
      (
      echo ">>>>>>>>>>>>> Now testing ${experiment_tag}"

      CUDA_VISIBLE_DEVICES=$device_id python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns 'Monthly' \
      --model_id m4_Monthly \
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
      --d_model 32 \
      --d_ff 32 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --dBG $arch \
      --k $k_param \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag 

    CUDA_VISIBLE_DEVICES=$device_id python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns 'Yearly' \
      --model_id m4_Yearly \
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
      --d_model 16 \
      --d_ff 32 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --dBG $arch \
      --k $k_param \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag 

    CUDA_VISIBLE_DEVICES=$device_id python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns 'Quarterly' \
      --model_id m4_Quarterly \
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
      --d_model 64 \
      --d_ff 64 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --dBG $arch \
      --k $k_param \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag 

    CUDA_VISIBLE_DEVICES=$device_id python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns 'Daily' \
      --model_id m4_Daily \
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
      --d_model 16 \
      --d_ff 16 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --dBG $arch \
      --k $k_param \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag 

    CUDA_VISIBLE_DEVICES=$device_id python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns 'Weekly' \
      --model_id m4_Weekly \
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
      --d_model 32 \
      --d_ff 32 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --dBG $arch \
      --k $k_param \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag 

    CUDA_VISIBLE_DEVICES=$device_id python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns 'Hourly' \
      --model_id m4_Hourly \
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
      --d_model 32 \
      --d_ff 32 \
      --top_k 5 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --dBG $arch \
      --k $k_param \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag 
      ) | tee -a "logs/${experiment_tag}.txt"
}

for arch in "${architectures[@]}"; do
  for k_param in "${k_params[@]}"; do
    for disc in "${disc_params[@]}"; do
      if [[ "$arch" == "graph_emb" ]]; then
        for dbgdim in "${graph_dim_params[@]}"; do
          feat_count=-1
          include_corr=False
          run_experiment
        done
      else
        for feat_count in "${motif_count_params[@]}"; do
          for include_corr in "${include_corr_params[@]}"; do
            dbgdim=-1
            run_experiment
          done
        done
      fi
    done
  done
done  
        