
# 说明
v2 版本使用MCdropout 预测
   0  individual
   1  FedMD
   2  使用信息量
   3  使用不确定度
   
# run 

        python3 main.py --dataset cifar10 --epochs 200 --num_users 10 --local_ep 1 --local_bs 64 --policy 2  --train_num 1000 --name fedavg
        
        policy = 1, seperate traing, policy >= 2 fed traing 
        col_policy = 1-mutual learning  2-average
        use_all_data = 0-normal split  1-all_clients have all data
## 使用公共数据集
        --col_policy = 1 从avg_label 学习,=2 mutual learning
        --pub_data 公共数据集的名称
        python3 main.py --dataset cifar10 --epochs 200 --num_users 10 --local_ep 1 --local_bs 64 --policy 2  --train_num 1000 --col_policy 1 --pub_data cifar100 --name use_pub_data
## noniid 部分
        --iid 1 --noniid pathological --alpha 2
## 互学习部分
        --col_policy 1 --pub_data cifar10 --pub_data_num 5000
# 一些实例
## 正常单独训练，iid
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 64 --train_num 1000 --policy 1  
## 正常联邦学习，iid
        python3 main.py python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 64 --train_num 1000 --policy 2

## 单独训练-noniid
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 64 --train_num 1000  \
        --policy 1 \
        --iid 0 --noniid pathological --alpha 2  \
        --name individual_nod_iid
## fedavg-noniid
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 64 --train_num 1000\
        --policy 2   \
        --iid 0 --noniid pathological --alpha 2 \
        --name fedavg_non_iid

## fedmu-noniid
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 64 --train_num 1000\
        --policy 1   \
        --iid 0 --noniid pathological --alpha 2 \
        --col_policy 1 \
        --name fedmu_non_iid_2
## fedmd-noniid
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 64 --train_num 1000\
        --policy 1   \
        --iid 0 --noniid pathological --alpha 2 \
        --col_policy 2 \
        --name fedmu_non_iid_2


##       mutual learning 
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 300 --num_users 4 --local_ep 0 --local_bs 64 --train_num 1000 \   
        --optimizer StepLR --step_size 60 --lr 0.1 \
        --policy 1   \
        --col_policy 1 \
        --pub_data cifar100 --pub_data_num 60000 --pub_data_labeled 1 \
        --name my_mutual
##       average
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 300 --num_users 4 --local_ep 0 --local_bs 64 --train_num 1000\
        --policy 1   \
        --col_policy 2 \
        --pub_data cifar100 --pub_data_num 60000 --pub_data_labeled 1 \
        --name my_average

        launch -- \
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --name fedmd_avg_0_alpha_8

        launch -- \
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 1 \
        --name fedmd_avg_1_alpha_8


        launch -- \
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 \
        --name fedmd_avg_2_alpha_8
        
##       change weight_temperature
        launch -- \
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 2 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 --weight_temperature 0.1 \
        --name fedmd_avg_2_alpha_2_weight_temperature_0.1
        

## cifar100 测试
        launch -- \
        python3 main.py  --dataset cifar100 --num_classes 100 -- --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --name fedmd_avg_0_alpha_8

        launch -- \
        python3 main.py --dataset cifar100 --num_classes 100 --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 1 \
        --name fedmd_avg_1_alpha_8


        launch -- \
        python3 main.py --dataset cifar10 --epochs 300 --num_users 4 --local_ep 1 --local_bs 256 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid pathological --alpha 8 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 1000 --use_avg_loss 2 \
        --name fedmd_avg_2_alpha_8
        
## 全连接神经网络
