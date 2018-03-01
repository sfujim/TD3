#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
	python2 main.py \
	--policy_name "TD3" \
	--env_name "HalfCheetah-v1" \
	--seed $i \
	--start_timesteps 10000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Hopper-v1" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Walker2d-v1" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Ant-v1" \
	--seed $i \
	--start_timesteps 10000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "InvertedPendulum-v1" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "InvertedDoublePendulum-v1" \
	--seed $i \
	--start_timesteps 1000

	python2 main.py \
	--policy_name "TD3" \
	--env_name "Reacher-v1" \
	--seed $i \
	--start_timesteps 1000
done
