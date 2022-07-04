# 验证 avg 和 weight avg

cifar10 
        pathological non-iid client = 10 ,train_num = 1000,epcoh = 500
        dirichlet chilent = 4, train_num = 5000, alpha = 0.1,1,100

cifar100 
        dirichlet client = 5, train_num = 5000,10000
##      cifar10
###     pathological
####     0
        python3 main.py --dataset cifar100 --num_classes 10 --epochs 500 --num_users 10 --local_ep 0 --local_bs 64 --train_num 1000\
        --policy 1   \
        --col_policy 2 \
        --iid 0 --noniid pathological --alpha 8 \
        --name my_average
####     1
        launch -- \
        python3 main.py --dataset cifar10 --epochs 500 --num_users 10 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3 \
        --name fedmd_avg_0_alpha_3
####     2
        launch -- \
        python3 main.py --dataset cifar10 --epochs 500 --num_users 10 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 1 \
        --name fedmd_avg_1_alpha_3

####    3
        launch -- \
        python3 main.py --dataset cifar10 --epochs 500 --num_users 10 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 5 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2    \
        --name fedmd_avg_2_alpha_5
####    4 滑动平均
        launch -- \
        python3 main.py --dataset cifar10 --epochs 1000 --num_users 10 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 2 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 --ema 0.9   \
        --name fedmd_avg_2_alpha_2_ema_0.9
####.   5. 卡尔曼滤波置信度计算
        launch -- \
        python3 main.py --dataset cifar10 --epochs 500 --num_users 10 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 10 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 --kalman 1   \
        --name fedmd_avg_2_alpha_10_kalman_3
###      change weight_temperature
        launch -- \
        python3 main.py --dataset cifar10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 3 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 --weight_temperature 0.5 \
        --name fedmd_avg_2_alpha_3_weight_temperature_0.5
        

###     dirichlet
0.1,0.5,1,100
####     0
        launch -- \
        python3 main.py  --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha $1 \
        --name cifar10_fedmd_avg_0_alpha_d_$1
####     1
        launch -- \
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha $1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 5000 --use_avg_loss 1 \
        --name cifar10_fedmd_avg_1_alpha_d_$1
####     2 
        launch  -- \
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha  $1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 5000 --use_avg_loss 2 \
        --name cifar10_fedmd_avg_2_alpha_d_$1
 ####     3 kalman1
        launch  -- \
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha $1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 5000 --use_avg_loss 2 --kalman 1 \
        --name cifar10_fedmd_avg_2_alpha_d_$1_kalman_1
####     4 kalman2
        launch  -- \
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha $1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 5000 --use_avg_loss 2 --kalman 2 \
        --name cifar10_fedmd_avg_2_alpha_d_$1_kalman_2  


##      cifar100 测试
###     iid 
####    0
        launch -- \
        python3 main.py  --dataset cifar100 --num_classes 100 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 10000 \
        --policy 1   \
        --name cifar100_fedmd_avg_0_iid
####     1
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 10000 \
        --policy 1   \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 10000 --use_avg_loss 1 \
        --name cifar100_fedmd_avg_1_iid
####     2
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 10000 \
        --policy 1   \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 10000 --use_avg_loss 2 \
        --name cifar100_fedmd_avg_2_iid
###     non-iid
1,5,10,50,100
####     0
        launch -- \
        python3 main.py  --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --name cifar100_fedmd_avg_0_alpha_d_0.1
####     1
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 5000 --use_avg_loss 1 \
        --name cifar100_fedmd_avg_1_alpha_d_0.1
####     2
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 5000 --use_avg_loss 2 \
        --name cifar100_fedmd_avg_2_alpha_d_0.1
####   3 kalman
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.1 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 5000 --use_avg_loss 2 --kalman 1 \
        --name cifar100_fedmd_avg_2_alpha_d_$1_kalman_0.1    

Pathological 
####     0
        launch -- \
        python3 main.py  --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 60 \
        --name cifar100_fedmd_avg_0_alpha_60
####     1
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 80 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 5000 --use_avg_loss 1 \
        --name cifar100_fedmd_avg_1_alpha_80
####     2
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 80 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 5000 --use_avg_loss 2 \
        --name cifar100_fedmd_avg_2_alpha_80
####   3 kalman
        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 80 \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 5000 --use_avg_loss 2 --kalman 1 \
        --name cifar100_fedmd_avg_2_alpha_80_kalman_1              
### fedavg + distillation


##      mmist 测试
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple filterpy 
        python3 main.py --dataset mnist --num_classes 100 --epochs 500 --num_users 5 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 10 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 5000 --use_avg_loss 3 \
        --name cifar100_fedmd_avg_2_alpha_80