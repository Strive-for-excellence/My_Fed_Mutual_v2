import copy
import torch
# import numpy as np
# from utils.load_dataset import digits_dataset_read, digits_dataset_read_test,digits_dataset_loader, PairedData
from src.client import Client
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable
import numpy as np
from filterpy.stats import multivariate_multiply
class Server:
    '''
    local_model is the model architecture of each client;
    global_model is the model need to be aggregated;
    '''
    def __init__(self, device, global_model,clients, args, pub_data=''):
        self.device = device
        self.global_model = global_model
        self.args = args
        self.total_clients =  self.args.num_users
        # indexes set of clients
        self.indexes = [i for i in range(self.total_clients)]
        # 获取全局数据集
        # self.get_global_dataset(domains, args)
        # 生成用户
        self.pub_data = pub_data
        self.clients = clients
        self.send_parameters()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.pre_result = []
    # def get_model(self):
    #
    #     client_model = []
    #     self.global_model = ResNet101(self.args.class_nums)
    #     if self.args.policy == 3:
    #         half = self.args.num_users//2
    #         for i in range(half):
    #             client_model.append(copy.deepcopy(self.global_model))
    #         for i in range(half,self.args.num_users):
    #             client_model.append(ResNet(Bottleneck, [3, 4, 22, 3], num_classes=self.args.num_classes))
    #
    #     else :
    #         for i in range(self.args.num_users):
    #             client_model.append(copy.deepcopy(self.global_model))
    #
    #     return client_model
    #     # for i in range(self.args.num_users):
    #     # self.send_parameters(model.state_dict())/

    def fit_multivariate_gaussian_distribution(self,X):
        mean = np.mean(X, axis=0)
        cov = np.cov(X,rowvar=0)
        return mean,cov
    def average_weights(self, w):
        '''
        :param w: weights of each client
        :return: the average of the weights
        '''
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            cnt = 1
            for client in range(1,self.args.num_users):
                if key in w[client].keys():
                    w_avg[key] += w[client][key]
                    cnt += 1
            w_avg[key] = torch.div(w_avg[key],  cnt)
        return w_avg

    def get_parameters(self):
        local_weights = []
        for client in range(self.args.num_users):
            local_weights.append(copy.deepcopy(self.clients[client].local_model.state_dict()))
        return local_weights

    def send_parameters(self):
        w_avg = self.global_model.state_dict()
        if self.args.policy == 1:   # separate training
            return
        elif self.args.policy >= 2: # collaborate train a global model
            for client in range(self.args.num_users):
                local_model = self.clients[client].local_model.state_dict()
                for key in local_model.keys():
                    local_model[key] = w_avg[key]
                self.clients[client].local_model.load_state_dict(local_model)
            return
    def col_train(self):
        # with tqdm(total=self.args.pub_data_num) as pbar:
        if 1:
            for batch_idx, (images, labels) in enumerate(self.pub_data):
                # if self.pre_result:

                # for i in range(len(self.pre_result)):
                data_num = images.shape[0]
                # pbar.update(images.shape[0])
                # print(batch_idx)
                # if batch_idx * self.args.local_bs > 1000:
                #     break
                outputs = []
                images, labels = images.to(self.device), labels.to(self.device)
                images, labels = Variable(images),Variable(labels)
                # with torch.no_grad():

                #    use_avg_loss = 1
                if self.args.use_avg_loss == 1:
                    with torch.no_grad():
                        for i in range(self.args.num_users):
                            # self.clients[i].local_model.train()
                            outputs.append(self.clients[i].local_model(images))
                    avg_soft_label = Variable(F.softmax(sum(outputs) / len(outputs), dim=1))
                # 计算weight soft label entropy based
                elif self.args.use_avg_loss == 2:
                    with torch.no_grad():
                        for i in range(self.args.num_users):
                            # self.clients[i].local_model.train()
                            outputs.append(self.clients[i].local_model(images))
                    # softmax
                    outputs_tmp = [F.softmax(Variable(outputs[i]),dim=1).cpu().numpy() for i in range(self.args.num_users)]

                    # for i in range(self.args.num_users):
                    #     outputs_tmp[i] =
                    outputs_entropy = []

                    if self.args.kalman==0:
                        for i in range(self.args.num_users):
                            outputs_entropy.append(
                                np.log2(self.args.num_classes) - np.sum(-outputs_tmp[i] * np.log2(outputs_tmp[i]),
                                                                        axis=1))
                        all_entropy = np.stack(outputs_entropy, axis=0)
                        all_entropy = torch.tensor(all_entropy)/self.args.weight_temperature
                        all_entropy = F.softmax(all_entropy,dim=0)
                        all_entropy = torch.unsqueeze(all_entropy,-1)
                        # if batch_idx == 1:
                        #     print(all_entropy)
                        avg_soft_label = np.sum(np.array(outputs_tmp) * all_entropy.numpy(), axis=0)
                        avg_soft_label = torch.tensor(avg_soft_label)

                    else:
                        # kalman
                        for i in range(self.args.num_users):
                            outputs_entropy.append(np.sum(-outputs_tmp[i] * np.log2(outputs_tmp[i]),axis=1))
                        all_entropy = np.stack(outputs_entropy, axis=0)
                        sigma_divided_by_1  = 1/torch.square(torch.tensor(all_entropy))
                        sum_sigma = 1 / torch.sum(sigma_divided_by_1,dim=0)
                        weight = sum_sigma * torch.tensor(sigma_divided_by_1)
                        if self.args.kalman == 2:
                            weight = weight/self.args.weight_temperature
                            weight = F.softmax(weight,dim=0)
                        weight = torch.unsqueeze(weight,-1)
                        avg_soft_label =  torch.sum(torch.tensor(outputs_tmp)*weight,dim=0)
                    avg_soft_label = avg_soft_label.to(self.device)
                # 使用MC dropout uncertain based
                elif self.args.use_avg_loss == 3:
                    client_item_mean, client_item_cov = [],[]
                    for client in range(self.args.num_users):
                        self.clients[client].local_model.train()
                        with torch.no_grad():
                            times = 100
                            results = []
                            for time in range(times):
                                result = F.softmax(self.clients[client].local_model(images),dim=1)
                                results.append(result.cpu().numpy())
                                # tmp = torch.sum(result,dim=1)
                            # for item in len()
                            # 对每张图片处理
                            item_mean,item_cov = [],[]
                            for item in range(data_num):
                                temp = []
                                for time in range(times):
                                    temp.append(results[time][item])
                                temp = np.array(temp)
                                mean,cov = self.fit_multivariate_gaussian_distribution(temp)
                                item_mean.append(mean)
                                item_cov.append(cov)
                            client_item_mean.append(item_mean)
                            client_item_cov.append(item_cov)
                    # 融合
                    item_mean ,item_cov  = [],[]
                    for item in range(data_num):
                        tmp_mean = client_item_mean[0][item]
                        eps = 1e-12
                        tmp_cov = client_item_cov[0][item]+np.eye(self.args.num_classes)*eps
                        for client in range(1,self.args.num_users):
                            tmp_mean,tmp_cov = multivariate_multiply(tmp_mean,tmp_cov,
                                                                     client_item_mean[client][item],client_item_cov[client][item]+np.eye(self.args.num_classes)*eps)
                        item_mean.append(tmp_mean)
                        item_cov.append(tmp_cov)
                    item_mean = np.array(item_mean)
                    avg_soft_label = torch.FloatTensor(item_mean)
                elif self.args.use_avg_loss == 4:
                    client_item_mean, client_item_cov = [],[]
                    for client in range(self.args.num_users):
                        self.clients[client].local_model.train()
                        with torch.no_grad():
                            times = 100
                            results = []
                            for time in range(times):
                                result = F.softmax(self.clients[client].local_model(images),dim=1)
                                results.append(result.cpu().numpy())
                                # tmp = torch.sum(result,dim=1)
                            # for item in len()
                            # 对每张图片处理
                            item_mean,item_cov = [],[]
                            for item in range(data_num):
                                temp = []
                                for time in range(times):
                                    temp.append(results[time][item])
                                temp = np.array(temp)
                                mean,cov = self.fit_multivariate_gaussian_distribution(temp)
                                item_mean.append(mean)
                                item_cov.append(cov)
                            client_item_mean.append(item_mean)
                            client_item_cov.append(item_cov)
                    outputs = client_item_mean
                    # softmax
                    outputs_tmp = np.array(outputs)
                    outputs_entropy = []
                    if self.args.kalman==0:
                        for i in range(self.args.num_users):
                            outputs_entropy.append(
                                np.log2(self.args.num_classes) - np.sum(-outputs_tmp[i] * np.log2(outputs_tmp[i]),
                                                                        axis=1))
                        all_entropy = np.stack(outputs_entropy, axis=0)
                        all_entropy = torch.tensor(all_entropy)/self.args.weight_temperature
                        all_entropy = F.softmax(all_entropy,dim=0)
                        all_entropy = torch.unsqueeze(all_entropy,-1)
                        # if batch_idx == 1:
                        #     print(all_entropy)
                        avg_soft_label = np.sum(np.array(outputs_tmp) * all_entropy.numpy(), axis=0)
                        avg_soft_label = torch.tensor(avg_soft_label)

                    else:
                        # kalman
                        for i in range(self.args.num_users):
                            outputs_entropy.append(np.sum(-outputs_tmp[i] * np.log2(outputs_tmp[i]),axis=1))
                        all_entropy = np.stack(outputs_entropy, axis=0)
                        sigma_divided_by_1  = 1/torch.square(torch.tensor(all_entropy))
                        sum_sigma = 1 / torch.sum(sigma_divided_by_1,dim=0)
                        weight = sum_sigma * torch.tensor(sigma_divided_by_1)
                        if self.args.kalman == 2:
                            weight = weight/self.args.weight_temperature
                            weight = F.softmax(weight,dim=0)
                        weight = torch.unsqueeze(weight,-1)
                        avg_soft_label =  torch.sum(torch.tensor(outputs_tmp)*weight,dim=0)
                    avg_soft_label = avg_soft_label.to(self.device)
                    # for i in range(self.args.num_users):
                    #     outputs_tmp[i] =
                    outputs_entropy = []
                avg_soft_label = avg_soft_label.to(self.device)
                sum_avg = torch.sum(avg_soft_label,dim=1)
                # 利用无标签数据集训练
                for i in range(self.args.num_users):

                    # predict = outputs[i]
                    self.clients[i].local_model.train()
                    predict = self.clients[i].local_model(images)

                    # 如果需要滑动，默认args.ema=0,不需要
                    if len(self.pre_result) > batch_idx:
                        self.pre_result[batch_idx] = self.pre_result[batch_idx]*self.args.ema + avg_soft_label *(1-self.args.ema)
                    else:
                        self.pre_result.append(avg_soft_label)
                    loss = self.loss_kl(F.log_softmax(predict,dim=1),self.pre_result[batch_idx])

                    self.clients[i].optimizer.zero_grad()
                    loss.backward()
                    self.clients[i].optimizer.step()

    def train(self):
        test_losses = []
        test_acc = []

        # local_weights = []
        for epoch in tqdm(range(self.args.epochs)):
            print(f'Start Training round: {epoch}')
            if self.args.optimizer == 'StepLR':
                for client in range(self.args.num_users):
                    self.clients[client].scheduler.step()
            print(f'lr: {self.clients[0].optimizer.param_groups[0]["lr"]}')
            # test_losses.clear()
            # test_acc.clear()
            # 模块1 训练
            for client in range(self.args.num_users):
                print('client = ',client)
                self.clients[client].train()
            # 模块2 知识蒸馏
            # select clients to train their local model
            # 获得模型对公共数据集的预测
            if self.args.col_policy != 0:
                # pub_predict_list = []
                # for client in range(self.args.num_users):
                #     pub_predict_list.append(self.clients[client].predict_pub_data(self.pub_data))
                # for client in range(self.args.num_users):
                #     self.clients[client].train_on_pub_data(self.pub_data,client,pub_predict_list)
                for e in range(self.args.col_epoch):
                    self.col_train()


            # 模块3 预测
            local_test_losses = []
            local_test_acc = []


            # test on each clients

            for client in range(self.args.num_users):
                acc, loss = self.clients[client].inference()
                print('client = ',client,' acc = ',acc,' loss = ',loss)
                local_test_acc.append(copy.deepcopy(acc))
                local_test_losses.append(copy.deepcopy(loss))
            test_losses.append(sum(local_test_losses)/len(local_test_losses))
            test_acc.append(sum(local_test_acc)/len(local_test_acc))

            # # send parameters to each client
            # self.send_parameters(w_avg)

            # print the training information in this epoch
            print(f'\nCommunication Round: {epoch}   Policy: {self.args.policy}')
            print(f'Avg testing Loss: {test_losses[-1]}')
            print(f'Avg test Accuracy: {test_acc[-1]}')

            self.test_losses = test_losses
            self.test_acc = test_acc
            # if self.args.early_stop and len(test_losses)>100:
            #   if min(test_losses[0:-50]) < min(test_losses[-50:]):
            #       break
            if epoch%10 == 0:
                self.save_result()

        self.test_losses = test_losses
        self.test_acc = test_acc
        self.save_result()
        self.print_res()
        return

    def print_res(self):
        print(f'Final Accuracy :{self.test_acc[-1]}')
        print(f'Best Accuracy:{max(self.test_acc)}')
        # for domain in self.domains:
        #     print(f'domain: {domain}')
        #     print(f'Best accuracy: {max(self.domain_test_acc[domain])}')

    def save_result(self):
        # import shelve
        # from contextlib import closing
        # with closing(shelve.open(f'./save/Result_R({self.args.epochs})_'
        #     f'N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})_name({self.args.name}'
        #                          f'_P({self.args.policy})','c')) as shelve:
        #     shelve['test_losses'] = self.test_losses
        #     shelve['test_acc'] = self.test_acc
        import json
        json_name = f'./save/Result'\
                f'_dataset({self.args.dataset})'\
                f'_R({self.args.epochs})'\
                f'_N({self.args.num_users})_E({self.args.local_ep})_trainnum({self.args.train_num})'\
                f'_P({self.args.policy})_name({self.args.name}).json'
        # print('json_name = ',json_name)
        with open(json_name,mode='w+') as f:
            result = {}
            result['test_losses'] = self.test_losses
            result['test_acc'] = self.test_acc
            json.dump(result,f)
        print(json_name)
    def save_model(self,state='before_finetune'):
        for client in range(self.args.num_users):
            # acc, loss = self.clients[domain][client].inference()
            torch.save(self.clients[client].local_model,'./save/'+state+'.model')
