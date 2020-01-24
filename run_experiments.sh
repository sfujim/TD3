#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=6

for ((i=3;i<10;i+=1))
do 

  # model free
  python main.py \
  --max_timesteps 1000000 \
  --policy_name "TD3" \
  --env_name "Hopper-v2" \
  --seed $i \
	--save_models &

	# backward model based
	python main.py \
	--save_models \
  --max_timesteps 1000000 \
  --policy_name "TD3" \
  --bwd_model_update_freq 5e3 \
  --env_name "Hopper-v2" \
  --seed $i \
  --model_based backward \
  --model_iters 10 &

  # forward model based
  python main.py \
	--save_models \
  --max_timesteps 1000000 \
  --policy_name "TD3" \
  --fwd_model_update_freq 5e3 \
  --env_name "Hopper-v2" \
  --seed $i \
  --model_based forward \
  --model_iters 10 &

done

wait