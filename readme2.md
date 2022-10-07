# v2 版本使用MCdropout 预测
   0  individual
   1  FedMD
   2  使用信息量
   3  使用不确定度


##      mmist 测试
        0
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3  \
        --name mnist_fedmd_avg_0_alpha_3

        1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 1  \
        --name mnist_fedmd_avg_1_alpha_3

        2
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2  \
        --name mnist_fedmd_avg_2_alpha_3

        2_kalman_1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 --kalman 1 \
        --name mnist_fedmd_avg_2_alpha_3_kalman_1

        4
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 4 \
        --name mnist_fedmd_avg_4_alpha_3

       4 kalman 1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 10 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 4 \
        --name mnist_fedmd_avg_4_alpha_10_kalman_1

        5 基于 max {p_c} c \in [1,numclasses]
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 10 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 5  \
        --name mnist_fedmd_avg_5_alpha_10
# 狄利克雷non-iid
        0
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.005 \
        --name mnist_fedmd_avg_0_d_alpha_0.005

        1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.005 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 1 \
        --name mnist_fedmd_avg_1_d_alpha_0.005

        2
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.005 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 \
        --name mnist_fedmd_avg_2_d_alpha_0.005

        2_kalman_1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.005 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 --kalman 1 \
        --name mnist_fedmd_avg_2_d_alpha_0.005_kalman_1

        4
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.005 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 4  \
        --name mnist_fedmd_avg_4_d_alpha_0.005

        4_kalman_1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.005 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 4 --kalman 1 \
        --name mnist_fedmd_avg_4_d_alpha_0.005_kalman_1

        5 基于 max {p_c} c \in [1,numclasses]
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 5  \
        --name mnist_fedmd_avg_5_d_alpha_0.01

## cifar10

###     dirichlet
1,0.5,1,100
####     0
        python3 main.py  --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 100 \
        --name cifar10_fedmd_avg_0_alpha_d_100
####     1
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 100 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 1 \
        --name cifar10_fedmd_avg_1_alpha_d_100
####     2 
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 --lr 0.001\
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha  1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 \
        --name cifar10_fedmd_avg_2_alpha_d_100
 ####     2 kalman1
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 100 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 --kalman 1 \
        --name cifar10_fedmd_avg_2_alpha_d_100_kalman_1
####     4 
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 100 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 4  \
        --name cifar10_fedmd_avg_4_alpha_d_100
####     4 kalman1
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 100 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 4 --kalman 1 \
        --name cifar10_fedmd_avg_4_alpha_d_100_kalman_1  
####    5
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 100 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 5 \
        --name cifar10_fedmd_avg_5_alpha_d_100 

## cifar100 
###     non-iid
1,5,10,50,100
####     0
        python3 main.py  --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --name cifar100_fedmd_avg_0_alpha_d_0.1
####     1
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 1000 --use_avg_loss 1 \
        --name cifar100_fedmd_avg_1_alpha_d_0.1
####     2
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 1000 --use_avg_loss 2 \
        --name cifar100_fedmd_avg_2_alpha_d_0.1
####   2 kalman_1
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 1000 --use_avg_loss 2 --kalman 1 \
        --name cifar100_fedmd_avg_2_alpha_d_0.1_kalman_1 
####     4
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 1000 --use_avg_loss 4 \
        --name cifar100_fedmd_avg_4_alpha_d_0.1
####   4 kalman_1
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 1000 --use_avg_loss 4 --kalman 1 \
        --name cifar100_fedmd_avg_4_alpha_d_0.1_kalman_1