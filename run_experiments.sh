#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
	python3.6 main.py \
	--policy "TD3" \
	--env "HalfCheetah-v3" \
	--seed $i

	python3.6 main.py \
	--policy "TD3" \
	--env "Hopper-v3" \
	--seed $i

	python3.6 main.py \
	--policy "TD3" \
	--env "Walker2d-v3" \
	--seed $i

	python3.6 main.py \
	--policy "TD3" \
	--env "Ant-v3" \
	--seed $i

	python3.6 main.py \
	--policy "TD3" \
	--env "Humanoid-v3" \
	--seed $i

	python3.6 main.py \
	--policy "TD3" \
	--env "InvertedPendulum-v2" \
	--seed $i \
	--start_timesteps 1000

	python3.6 main.py \
	--policy "TD3" \
	--env "InvertedDoublePendulum-v2" \
	--seed $i \
	--start_timesteps 1000

	python3.6 main.py \
	--policy "TD3" \
	--env "Reacher-v2" \
	--seed $i \
	--start_timesteps 1000
done
