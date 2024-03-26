# With dBG

alias python='venv/bin/python3.8'
model_name=TimesNet

seasonal_patterns=("Weekly" "Daily" "Quarterly" "Hourly" "Yearly" "Monthly")
d_models=(32)

for d_model in "${d_models[@]}"; do
  echo "!!!! Now testing :${d_model}" | tee -a out.txt
    for pattern in "${seasonal_patterns[@]}"; do
        python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path ./dataset/m4 \
            --seasonal_patterns $pattern \
            --model_id m4_$pattern \
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
            --d_model $d_model \
            --d_ff 32 \
            --top_k 5 \
            --des Exp \
            --itr 1 \
            --learning_rate 0.001 \
            --loss SMAPE \
            --train_epochs 10 \
            --dBG True | tee -a out.txt
    done
done