# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2022/04/02
@file: baseline.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
"""  
import torch
from torch import optim
from torch.utils.data import dataset
import os
from my_utils import *
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('--dataset', type=str, default='KLSG')
    parser.add_argument('--p_value', type=int, default=0)
    parser.add_argument('--k_value', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--save_results', type=str, default='True')
    parser.add_argument('--save_models', type=str, default='False')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get args
    args = parse_args()

    # get model
    dataset = args.dataset
    p_v = args.p_value
    k_v = args.k_value
    backbone = args.backbone
    if dataset == 'KLSG':
        nb_classes = 2
    elif dataset == 'LTSID':
        nb_classes = 8
    elif dataset == 'FLSMDD':
        nb_classes = 10
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')
    
    if backbone in ['vgg16', 'vgg19']:
        lr = 0.005
    elif backbone in ['resnet18', 'resnet34', 'resnet50']:
        lr = 0.01
    else:
        print(f'ERROR! BACKBONE {backbone} IS NOT EXIST!')
    
    is_train = True
    try:
        model = get_pretrained_model(backbone, is_train)
    except:
        print(f'THE INPUT BACKBONE {backbone} IS NOT EXIST!')
    model = model_fc_fix(model, nb_classes)
    # print(model)

    # get data iter
    sample_type = 'uniform'
    batch_size = 32
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../data', dataset) 
    kfold_train_idx, kfold_val_idx = get_kfold_img_idx(p=p_v, k=k_v, dataset=dataset, sample_type=sample_type)
    train_iter, val_iter = get_kfold_img_iters(batch_size, data_dir, kfold_train_idx, kfold_val_idx, mean, std)

    # set loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    output_params = list(map(id, model.last_linear.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, model.parameters())
    optimizer = optim.SGD([{'params': feature_params},
                        {'params': model.last_linear.parameters(), 'lr': lr * 10}],
                        lr=lr, weight_decay=0.001)

    # fine-tuning model
    num_epochs = 5
    train(train_iter, val_iter, model, loss, optimizer, device, num_epochs)

    # evaluating model
    gmean, y_hat, y_true, logits = evaluate_gmean(val_iter, model, device=device, if_get_y=False, if_get_logits=True)

    # save results
    if args.save_results in ['True', 'true']:
        save_dir = os.path.join(curr_dir, '../output/result', dataset, 'baseline', backbone)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_result_to_file(save_dir, y_hat, y_true, logits, p=p_v, k=k_v)

    # save models
    if args.save_models in ['True', 'true']:
        model_dir = os.path.join(curr_dir, f'../output/model/KLSG/p{p_v}_k{k_v}_{backbone}_baseline.pth')
        torch.save(model.state_dict(), model_dir)
