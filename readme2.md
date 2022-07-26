# v2 版本使用MCdropout 预测
   0  individual
   1  FedMD
   2  使用信息量
   3  使用不确定度


##      mmist 测试
        0
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --name mnist_fedmd_avg_0_alpha_8

        1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 1 \
        --name mnist_fedmd_avg_1_alpha_8_test2

        2
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 \
        --name mnist_fedmd_avg_2_alpha_8

        2_kalman_1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 \
        --name mnist_fedmd_avg_2_alpha_8_kalman_1

        3
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 3 \
        --name mnist_fedmd_avg_3_alpha_8_test2

# 狄利克雷non-iid
        0
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --name mnist_fedmd_avg_0_d_alpha_0.1

        1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 1 \
        --name mnist_fedmd_avg_1_d_alpha_0.1

        2
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 \
        --name mnist_fedmd_avg_2_d_alpha_0.1

        2_kalman_1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 2 \
        --name mnist_fedmd_avg_2_d_alpha_0.1_kalman_1

        3
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 5 --local_ep 1 --local_bs 100 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 100 --use_avg_loss 3 \
        --name mnist_fedmd_avg_3_d_alpha_0.1
