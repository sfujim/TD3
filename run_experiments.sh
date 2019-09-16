#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
	python main.py \
	--policy_name "TD3" \
	--env_name "HalfCheetah-v2" \
	--seed $i \
	--start_timesteps 10000

	python main.py \
	--policy_name "TD3" \
	--env_name "Hopper-v2" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--policy_name "TD3" \
	--env_name "Walker2d-v2" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--policy_name "TD3" \
	--env_name "Ant-v2" \
	--seed $i \
	--start_timesteps 10000

	python main.py \
	--policy_name "TD3" \
	--env_name "InvertedPendulum-v2" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--policy_name "TD3" \
	--env_name "InvertedDoublePendulum-v2" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--policy_name "TD3" \
	--env_name "Reacher-v2" \
	--seed $i \
	--start_timesteps 1000
done
