#!/bin/sh

declare -a files=(
  "DDPG.py"
  "main.py"
  "utils.py"
)

for i in "${files[@]}"
do
  gcloud compute scp --project product-ml --zone us-west1-b $i pytorch-vm:~/bpw/$i
done

echo "\nDONE."
