import torch
from utils.options import args_parser
import numpy as np
import copy
from src.client import Client
from src.server import Server
from utils.data import DataSet,get_pub_data,build_dataset
from model import *
def exp_parameter(args):
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number : {args.num_users}')
    print(f'Local Epochs: {args.local_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Policy: {args.policy}')


def get_model(args):
    client_model = []
    # global_model = resnet32(num_classes=args.num_classes)
    global_model = ResNet18(num_classes=args.num_classes)
    # mnist
    if args.dataset == 'mnist':
        global_model = DNN(1*28*28,10,1200)
    elif args.dataset == 'cifar10':
        global_model = LeNet_cifar10()
    else:
        global_model = ResNet18_drop(num_classes=args.num_classes)
    # if args.model in ['ResNet18','ResNet18','ResNet50','ResNet101','ResNet152','ResNet18_Sto','ResNet34_Sto','ResNet50_Sto','ResNet101_Sto','ResNet152_Sto']:
    #     if args.model == 'ResNet18':
    #         global_model = ResNet18(args.num_classes)
    #     if args.model == 'ResNet34':
    #         global_model = ResNet18(args.num_classes)
    #     if args.model == 'ResNet50':
    #         global_model = ResNet50(args.num_classes)
    #     if args.model == 'ResNet101':
    #         global_model = ResNet101(args.num_classes)
    #     if args.model == 'ResNet152':
    #         global_model = ResNet152(args.num_classes)
    for i in range( args.num_users):
        client_model.append(copy.deepcopy( global_model))
    return global_model,client_model

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # local_model = ResNet18(10)
    # print(f'Model Structure: {local_model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #  prepare data

    # prepare model
    # global_model = ResNet101(args.num_classes)

    global_model,client_model = get_model(args)
    if args.use_all_data == 0:
        dataset = DataSet(args)
        clients = [Client(device, client_model[i], dataset.train[i], dataset.test, args) for i in range(args.num_users)]
    else:
        train_set,test_set = build_dataset(args)
        train_set = torch.utils.data.DataLoader(train_set, batch_size = args.local_bs, shuffle=True)
        test_set = torch.utils.data.DataLoader(test_set, batch_size = args.local_bs,shuffle=False)
        clients = [Client(device, client_model[i], train_set, test_set, args) for i in range(args.num_users)]
    train_set,test_set = get_pub_data(args.pub_data,args)
    server = Server(device,global_model,clients,args,pub_data=train_set)
    server.train()
    # server.fine_fune()
    # server.train()
    server.print_res()
    server.save_result()


if __name__ == '__main__':


    args = args_parser()
    args.verbose = 0
    # set random seed
    np.random.seed(args.seed)
    exp_parameter(args)
    print(args)

    train(args)


