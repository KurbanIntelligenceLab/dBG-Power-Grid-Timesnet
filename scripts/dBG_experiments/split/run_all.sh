#!/bin/bash
nohup ./scripts/dBG_experiments/split/dbg_timesnet_append_linear.sh > /dev/null 2>&1 &
nohup ./scripts/dBG_experiments/split/dbg_timesnet_append.sh > /dev/null 2>&1 &
nohup ./scripts/dBG_experiments/split/dbg_timesnet_bypass.sh > /dev/null 2>&1 &
    nohup ./scripts/dBG_experiments/split/dbg_timesnet_graph_emb.sh > /dev/null 2>&1 &

