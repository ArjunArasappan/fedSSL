#!/bin/bash

# Define the range for num_clients and useResnet18 values
num_clients_values=(2 3 4 5 6)
useResnet18_values=(True False)

echo -n > filename

# Loop through each value of num_clients
for num_clients in "${num_clients_values[@]}"
do
  # Loop through each value of useResnet18
  for useResnet18 in "${useResnet18_values[@]}"
  do
    # Run the command with the current values of num_clients and useResnet18
    python main.py --num_clients=$num_clients --use_resnet18=$useResnet18
  done
done
