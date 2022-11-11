# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2022/04/02
@file: betl.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
"""                       
import torch
from torch import optim
import os
from my_utils import *
import pretrainedmodels as ptm
import pretrainedmodels.utils as utils
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='betl')
    parser.add_argument('--dataset', type=str, default='KLSG')
    parser.add_argument('--p_value', type=int, default=0)
    parser.add_argument('--k_value', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--save_prop', type=float, default=0.6)
    parser.add_argument('--save_results', type=str, default='True')
    parser.add_argument('--save_models', type=str, default='False')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get args
    args = parse_args()

    # set params
    dataset = args.dataset
    p_v = args.p_value
    k_v = args.k_value
    backbone = args.backbone
    if dataset == 'KLSG':
        nb_classes = 2
        subset_num = 6
    elif dataset == 'LTSID':
        nb_classes = 8
        subset_num = 19
    elif dataset == 'FLSMDD':
        nb_classes = 10
        subset_num = 7
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')

    if backbone in ['vgg16', 'vgg19']:
        feature_dim = 4096
        lr = 0.005
    elif backbone in ['resnet18', 'resnet34']:
        feature_dim = 512
        lr = 0.01
    elif backbone in ['resnet50']:
        feature_dim = 2048
        lr = 0.01
    else:
        print(f'ERROR! BACKBONE {backbone} IS NOT EXIST!')
    
    # --------- phase 1: transfer learning --------
    print('--------- phase 1: transfer learning --------')
    # get backbone model 
    is_train = True
    try:
        model = get_pretrained_model(backbone, is_train)
    except:
        print(f'THE INPUT BACKBONE {backbone} IS NOT EXIST!')
    model = model_fc_fix(model, nb_classes)

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

    # finetuning model
    num_epochs = 5
    train(train_iter, val_iter, model, loss, optimizer, device, num_epochs)

    # --------- phase 2: balanced ensemble learning --------
    print('--------- phase 2: balanced ensemble learning --------')
    # load pretrained model
    model.last_linear = ptm.utils.Identity() # remove the linear layer of pretrained backbone
    
    # load datasets
    balance_train_idxes = [get_kfold_img_idx(p=p_v, k=k_v, dataset=dataset, sample_type='balance')[0] for i in range(subset_num)]
    balance_train_iters, _ = get_kfold_img_iters(batch_size, data_dir, balance_train_idxes, kfold_val_idx, mean, std)

    # load fcs
    fcs = get_fcs(feature_dim, nb_classes, subset_num) 

    # balanced muti-branch training 
    lr2 = 0.01
    loss2 = torch.nn.CrossEntropyLoss()
    optimizer2 = []
    num_epochs2 = 5
    for i in range(len(fcs)):
        optimizer2.append(optim.SGD(fcs[i].parameters(), lr=lr2 * 10))
        train_mul(balance_train_iters[i], model, fcs[i], loss2, optimizer2[i], device, num_epochs2)

    # --------- phase 3: ensemble pruning --------
    print('--------- phase 3: ensemble pruning --------')
    save_prop = args.save_prop
    saved_fcs_idx = ensemble_pruning(train_iter, model, fcs, _p=save_prop)
    print(f'Saved fully connected layer indexes after ensemble pruning: {saved_fcs_idx}')
    saved_fcs = [fcs[i].eval() for i in saved_fcs_idx]

    # save results by weight_averaging after pruning
    if args.save_results in ['True', 'true']:
        _, y_hat, y_true, logits = evaluate_gmean_optional(val_iter, model, saved_fcs, 'weight_averaging', device=device, if_get_logits=True)
        save_dir = os.path.join(curr_dir, '../output/result', dataset, f'betl', backbone)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_result_to_file(save_dir, y_hat, y_true, logits, p=p_v, k=k_v)

    # save models
    if args.save_models in ['True', 'true']:
        fus_fc = get_fusion_fc(saved_fcs)
        model.last_linear = fus_fc
        model_folder = os.path.join(curr_dir, f'../output/model/{dataset}')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_dir = os.path.join(model_folder, f'p{p_v}_k{k_v}_{backbone}_baseline.pth')
        torch.save(model.state_dict(), model_dir)
