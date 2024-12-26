#!/bin/bash

SCRIPT_DIR="scripts/dBG_experiments/MW_Experiments"

# List of scripts to run in order (specify exact order)
scripts=(
    "Autoformer.sh"
    "Crossformer.sh"
    "DLinear.sh"
    "ETSFormer.sh"
    "FEDformer.sh"
    "FiLM.sh"
    "Informer.sh"
    "iTransformer.sh"
    "LightTS.sh"
    "MINC.sh"
    "Nonstationary_Transformer.sh"
    "Pyraformer.sh"
    "Reformer.sh"
    "TimesNet.sh"
    "Transformer.sh"
)


scripts2=(
  "FiLM.sh"
)

# Iterate and execute each script
for script in "${scripts2[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    if [ -x "$script_path" ]; then
        echo "Running $script..."
        bash "$script_path"
    else
        echo "Skipping $script (not executable or not found)."
    fi
done
