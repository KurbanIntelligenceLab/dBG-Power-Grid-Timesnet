#!/bin/bash

model_name=TimesNet
approximate=True
architectures=("graph_emb")
k_params=(4 5 6)
disc_params=(15 20 25)
graph_dim_params=(16 32 64 128)

run_experiment() {
      experiment_tag="A_${arch}_k${k_param}_d${disc}_ap${approximate}_gdim${dbgdim}_fdim${feat_count}_corr${include_corr}"
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
      --tag $experiment_tag \
      --use_multi_gpu
      )
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
      fi
    done
  done
done  
        