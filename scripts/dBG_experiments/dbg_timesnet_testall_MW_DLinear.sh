#!/bin/bash

model_name=DLinear
approximate=True
arch=graph_emb
k_params=(5 6)
disc_params=(15 20 25)
graph_dim_params=(16 32 64 128)
horizons=(24 48 96 168)

run_experiment() {
      experiment_tag="A_${arch}_k${k_param}_d${disc}_ap${approximate}_gdim${dbgdim}_fdim${feat_count}_corr${include_corr}_hor${horizon}"
      (
      echo ">>>>>>>>>>>>> Now testing ${experiment_tag}"

    python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/MW \
      --seasonal_patterns MW \
      --model_id MW \
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
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      --pred_len $horizon \
      --dBG $arch \
      --k $k_param \
      --ap $approximate \
      --disc $disc \
      --ap $approximate \
      --dBGEmb $dbgdim \
      --proto_feat $feat_count \
      --include_corr $include_corr \
      --tag $experiment_tag \
      --use_multi_gpu \
      --devices "0,1,2" \
      --train_epochs 30 \
      --patience 10
      )
}

for horizon in "${horizons[@]}"; do
  for k_param in "${k_params[@]}"; do
    for disc in "${disc_params[@]}"; do
      for dbgdim in "${graph_dim_params[@]}"; do
        feat_count=-1
        include_corr=False
        run_experiment
      done
    done
  done
done
