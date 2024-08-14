#!/bin/bash

# Define the range for num_clients and useResnet18 values

while kill -0 479311 2> /dev/null; do
    sleep 1
done


num_clients_values=(2 4 6 8 10)

echo -n > filename

# Loop through each value of num_clients
for num_clients in "${num_clients_values[@]}"
do
  python main.py --num_clients=$num_clients 
done
