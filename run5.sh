#!/usr/bin/env bash

SEED=20240226

mkdir -p test_logs

exp=$1
shift

for i in {0..4}; do
  logfile="test_logs/${exp}.$i.log"
  seed=$((SEED + i))
  python trainer.py -c config/$exp.yaml --seed_everything $seed "$@" 2>&1 >"$logfile"
  echo "$exp job $i"
  tail -7 "$logfile"
done



