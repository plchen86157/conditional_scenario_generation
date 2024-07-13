# bash script to record the training config

# for test the program on the complete dataset

# python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr 0.0015 -luf 50 -ldr 0.5 -we 300 -e 500 --adam_weight_decay 0.015 -w_p 5

# for test the program on the small dataset
# python train_tnt.py -d dataset/interm_data_small -o run/tnt/ -a -b 64 -c -cd 1 0 --lr 0.0010 -luf 10 -ldr 0.1



#!/bin/bash

# lr_values=(0.002)
# adam_weight_decay_values=(0.005 0.01 0.02)
# w_p_values=(75 100 125 150)

# lr_values=(0.001 0.002 0.004)
# adam_weight_decay_values=(0.01 0.02)
# w_p_values=(30 40 60)

# 3/21 best
# lr_values=(0.00025)
# adam_weight_decay_values=(0.01)
# w_p_values=(100)



# lr_values=(0.001 0.0001 0.00001)
# adam_weight_decay_values=(0.01 0.05)
# w_p_values=(10 30 50 100 400) #30
# n_s_values=(1 3 5 7 9 15 20) #30

# lr_values=(0.001 0.0001)
# adam_weight_decay_values=(0.01 0.05)
# n_s_values=(1 2 3 5 10 20) #(2 5 8 12 15 20) #30
# w_p_values=(20 30) #30

# 6/13 best
# lr_values=(0.001)
# adam_weight_decay_values=(0.01)
# n_s_values=(10)
# w_p_values=(30) 

# 6/17 best
# lr_values=(0.0005)
# adam_weight_decay_values=(0.01)
# n_s_values=(1 2)
# w_p_values=(29 30)

lr_values=(0.0005 0.001)
adam_weight_decay_values=(0.02 0.01)
n_s_values=(1 2) #(2 5 8 12 15 20) #30
w_p_values=(30 25) #30



for lr in "${lr_values[@]}"; do
    for adam_weight_decay in "${adam_weight_decay_values[@]}"; do
        for n_s in "${n_s_values[@]}"; do
            for w_p in "${w_p_values[@]}"; do
                # command="python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr $lr -luf 100 -ldr 0.5 -we 1 -e 2 --adam_weight_decay $adam_weight_decay -w_p $w_p"
                command="python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr $lr -luf 50 -ldr 0.75 -we 300 -e 500 --adam_weight_decay $adam_weight_decay -w_p $w_p -n_s $n_s"
                echo "Running command: $command"
                $command
            done
        done
    done
done

# lr_values=(0.00001)
# adam_weight_decay_values=(0.05 0.005)
# w_p_values=(100)

# for lr in "${lr_values[@]}"; do
#     for adam_weight_decay in "${adam_weight_decay_values[@]}"; do
#         for w_p in "${w_p_values[@]}"; do
#             # command="python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr $lr -luf 100 -ldr 0.5 -we 1 -e 2 --adam_weight_decay $adam_weight_decay -w_p $w_p"
#             command="python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr $lr -luf 100 -ldr 0.5 -we 300 -e 700 --adam_weight_decay $adam_weight_decay -w_p $w_p"
#             echo "Running command: $command"
#             $command
#         done
#     done
# done

# lr_values=(0.000005)
# adam_weight_decay_values=(0.01)
# w_p_values=(100)

# for lr in "${lr_values[@]}"; do
#     for adam_weight_decay in "${adam_weight_decay_values[@]}"; do
#         for w_p in "${w_p_values[@]}"; do
#             # command="python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr $lr -luf 100 -ldr 0.5 -we 1 -e 2 --adam_weight_decay $adam_weight_decay -w_p $w_p"
#             command="python train_tnt.py -o run/tnt/ -a -c -cd 0 -b 128 --lr $lr -luf 100 -ldr 0.5 -we 300 -e 700 --adam_weight_decay $adam_weight_decay -w_p $w_p"
#             echo "Running command: $command"
#             $command
#         done
#     done
# done