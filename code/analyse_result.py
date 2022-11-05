# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2022/04/02
@file: analyse_result.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
""" 
import os
import torch
import torch.nn.functional as F
from torch.utils.data import dataset
from my_utils import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve, roc_auc_score
from scipy import interpolate
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Get 10-trail 5-fold results from the saved files. 
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
    @ method : name of method, can be baseline or betl
    @ if_get_logits : if you want to get logits output, default is false
    @ trails : the number of trails results you want to get, default is 10
    @ folds : the number of folds results you want to get, default is 5
'''
def get_y_and_logits_results(dataset, backbone, method='baseline', if_get_logits=False, trails=10, folds=5):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone) 
    y_hat = []
    y_true = []
    logits = []
    for p in range(trails):
        for k in range(folds):
            if if_get_logits:
                y_h, y_t, lo = read_result_from_file(data_dir, p=p, k=k, if_get_logits=if_get_logits)
                logits = logits + lo
            else:
                y_h, y_t = read_result_from_file(data_dir, p=p, k=k, if_get_logits=if_get_logits)
            y_hat = y_hat + y_h
            y_true = y_true + y_t
    if if_get_logits:
        return y_true, y_hat, logits
    return y_true, y_hat

'''
Calculate Gmean result of each trail
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
    @ method : name of method, can be baseline or betl
    @ trails : the number of trails results you want to get, default is 10
    @ folds : the number of folds results you want to get, default is 5
'''
def get_gmean_each_trial(dataset, backbone, method='baseline', trails=10, folds=5):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone)
    gmean = [] 
    for p in range(trails):
        y_hat = []
        y_true = []
        for k in range(folds):
            y_h, y_t = read_result_from_file(data_dir, p=p, k=k)
            y_hat = y_hat + y_h
            y_true = y_true + y_t
        gmean.append(cal_gmean(y_true, y_hat))
    return gmean    

'''
Draw confusion matrix
    @ class_list : class name list, e.g., ['Plane', 'Wreck']
    @ y_true : the ground truth labels
    @ y_hat : the predict labels
    @ tail_idxes : indexes of tail classes, e.g., [0]
    @ pic_dir : confusion matrix picture save direction
'''
def draw_conf_matrix(class_list, y_true, y_hat, tail_idxes, pic_dir):
    conf_matrix = confusion_matrix(y_true, y_hat)
    prob_matrix = np.around((conf_matrix.T/np.sum(conf_matrix, 1)).T, 3)
    print('--------- Drawing confusion matrix ... ----------')
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(prob_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.)
    plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45, horizontalalignment='right', family='Times New Roman', fontsize=25)
    plt.yticks(tick_marks, class_list, family='Times New Roman', fontsize=25)
    for i in range(len(prob_matrix)):
        for j in range(len(prob_matrix)):
            if i in tail_idxes and j == i:
                plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', color='white', fontsize=20, fontweight='bold')
            elif j == i:
                plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', color='white', fontsize=20, fontweight='bold')
            else:
                plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', fontsize=17)
    for idx in tail_idxes:
        plt.gca().get_xticklabels()[idx].set_color('red')
        plt.gca().get_yticklabels()[idx].set_color('red')
    plt.tight_layout()
    plt.ylabel('True label', family='Times New Roman', fontsize=30, fontweight='bold')
    plt.xlabel('Predicted label', family='Times New Roman', fontsize=30, fontweight='bold')
    fig.savefig(pic_dir, format='pdf', bbox_inches='tight')
    print(f'Confusion matrix has been saved to {pic_dir}.')

'''
Transform label from [1, 2 ...] to one-hot format
    @ label : labels with original format, i.e., [1, 2, 3 ...]
    @ num_class : number of classes
'''
def label2onehot(label, num_class):
    label_tensor = torch.tensor(label)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    return label_onehot

'''
Draw F1 scale for Precision-Recall curves
'''
def draw_f1_scale():
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("F1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

'''
Draw macro average Precision-Recall curves of tail classes among different methods
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ num_class : number of classes
    @ tail_idxes : indexes of tail classes, e.g., [0]
    @ pic_dir : Precision-Recall curves picture save direction
'''
def draw_tail_classes_pr_curve(dataset, num_class, tail_idxes, pic_dir):
    print(f'--------- Drawing macro average P-R curves of tail classes ... ----------')
    # setup plot details    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['lightseagreen', 'red']

    # draw f1 scale
    draw_f1_scale()

    # set backbone and method list
    backbone = 'resnet18'
    method_list = ['baseline', 'betl']
    show_method_list = ['DTL', 'BETL(Ours)']
    
    for i in range(len(method_list)):
        y_true, y_hat, logits = get_y_and_logits_results(dataset, backbone, method_list[i], if_get_logits=True)
        if 'svm' in method_list[i]: 
            probs = logits
        else:
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        y_true = np.array(y_true)
        y_score = np.array(probs)
        score_array = np.array(y_score)

        # transform label to onehot format
        label_onehot = label2onehot(y_true, num_class)

        # calculate precision and recall corresponding to each class via sklearn
        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()
        inter_func_dict = dict()
        for j in range(num_class):
            precision_dict[j], recall_dict[j], _ = precision_recall_curve(label_onehot[:, j], score_array[:, j])
            inter_func_dict[j] = interpolate.interp1d(recall_dict[j], precision_dict[j], kind='linear')
            average_precision_dict[j] = average_precision_score(label_onehot[:, j], score_array[:, j])

        # calculate macro average precision and recall of tail classes
        all_r = np.unique(np.concatenate([recall_dict[k] for k in range(num_class)]))
        mean_p = np.zeros_like(all_r)
        for l in tail_idxes:
            mean_p += inter_func_dict[l](all_r)
        mean_p /= len(tail_idxes)
        recall_dict[f'macro_{method_list[i]}'] = all_r
        precision_dict[f'macro_{method_list[i]}'] = mean_p
        average_precision_dict[f'macro_{method_list[i]}'] = auc(all_r, mean_p)
   
        # draw macro average P-R curves of tail classes
        display = PrecisionRecallDisplay(
            recall=recall_dict[f'macro_{method_list[i]}'],
            precision=precision_dict[f'macro_{method_list[i]}'],
            average_precision=average_precision_dict[f'macro_{method_list[i]}'],
        )
        display.plot(ax=ax, name=f'{show_method_list[i]}', color=colors[i])

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])

    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_xlabel('Recall', fontsize=30, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=30, fontweight='bold')
    fig.savefig(pic_dir, format='pdf', bbox_inches='tight')
    print(f'Macro average P-R curves of tail classes have been saved to {pic_dir}.')

'''
Draw macro average Precision-Recall curves of all classes among different methods
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ num_class : number of classes
    @ pic_dir : Precision-Recall curves picture save direction
'''
def draw_macro_pr_curve(dataset, num_class, pic_dir):
    print(f'--------- Drawing macro average P-R curves of all classes ... ----------')
    # setup plot details    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['lightseagreen', 'red']

    # draw f1 scale
    draw_f1_scale()

    # set backbone and method list
    backbone = 'resnet18'
    method_list = ['baseline', 'betl']
    show_method_list = ['DTL', 'BETL(Ours)']
    for i in range(len(method_list)):
        y_true, y_hat, logits = get_y_and_logits_results(dataset, backbone, method_list[i], if_get_logits=True)
        if 'svm' in method_list[i]: 
            probs = logits
        else:
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        y_true = np.array(y_true)
        y_score = np.array(probs)
        score_array = np.array(y_score)

        # transform label to onehot format
        label_onehot = label2onehot(y_true, num_class)

        # calculate macro average precision and recall of all classes
        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()
        inter_func_dict = dict()
        for j in range(num_class):
            precision_dict[j], recall_dict[j], _ = precision_recall_curve(label_onehot[:, j], score_array[:, j])
            inter_func_dict[j] = interpolate.interp1d(recall_dict[j], precision_dict[j], kind='linear')
            average_precision_dict[j] = average_precision_score(label_onehot[:, j], score_array[:, j])
        
        # draw macro average P-R curves of all classes
        all_r = np.unique(np.concatenate([recall_dict[k] for k in range(num_class)]))
        mean_p = np.zeros_like(all_r)
        for l in range(num_class):
            mean_p += inter_func_dict[l](all_r)
        mean_p /= num_class
        recall_dict[f'macro_{method_list[i]}'] = all_r
        precision_dict[f'macro_{method_list[i]}'] = mean_p
        average_precision_dict[f'macro_{method_list[i]}'] = average_precision_score(label_onehot, score_array, average="macro")
            
        # draw macro average P-R curves of all classes
        display = PrecisionRecallDisplay(
            recall=recall_dict[f'macro_{method_list[i]}'],
            precision=precision_dict[f'macro_{method_list[i]}'],
            average_precision=average_precision_dict[f'macro_{method_list[i]}'],
        )
        display.plot(ax=ax, name=f'{show_method_list[i]}', color=colors[i])

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_xlabel('Recall', fontsize=30, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=30, fontweight='bold')
    fig.savefig(pic_dir, format='pdf', bbox_inches='tight')
    print(f'Macro average P-R curves of all classes have been saved to {pic_dir}.')

'''
Draw Precision-Recall curves of all classes 
    @ y_true : the ground truth labels
    @ logits: the predict scores of classification model
    @ class_list : class name list, e.g., ['Plane', 'Wreck']
    @ colors : color name list, e.g., ['orangered', 'lightseagreen']
    @ pic_dir : Precision-Recall curves picture save direction
'''
def draw_pr_curve(y_true, logits, class_list, colors, pic_dir):
    print(f'--------- Drawing P-R curves ... ----------')
    # setup plot details    
    fig, ax = plt.subplots(figsize=(10, 8))

    # draw f1 scale
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("F1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    y_true = np.array(y_true)
    y_score = np.array(logits)
    score_array = np.array(y_score)

    # transform label to onehot format
    label_onehot = label2onehot(y_true, num_class)

    # calculate precision and recall corresponding to each class
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])

    # draw P-R curves of each class
    for i, color in zip(range(num_class), colors):
        display = PrecisionRecallDisplay(
            recall=recall_dict[i],
            precision=precision_dict[i],
            average_precision=average_precision_dict[i],
        )
        display.plot(ax=ax, name=f"{class_list[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_xlabel('Recall', fontsize=30, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=30, fontweight='bold')
    fig.savefig(pic_dir, format='pdf', bbox_inches='tight')
    print(f'P-R curves have been saved to {pic_dir}.')

'''
Write Gmean result to file 
    @ gmean : Gmean result
    @ gmean_dir: Gmean save direction
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ method : name of method, can be baseline or betl
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
'''
def write_gmean(gmean, gmean_dir, dataset, method, backbone):
    print(f'--------- Writing Gmean results ... ----------')
    gmean_file = open(gmean_dir, 'a')
    gmean_file.write(f'[Time: {time.asctime()}] Gmean of {dataset}_{method}_{backbone} is : {gmean}\n')
    print(f'Gmean result: {gmean} have been saved to {gmean_dir}.')

'''
Write macro-F1 result to file 
    @ f1 : macro-F1 result
    @ f1_dir: macro-F1 save direction
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ method : name of method, can be baseline or betl
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
'''
def write_f1(f1, f1_dir, dataset, method, backbone):
    print(f'--------- Writing Macro-F1 results ... ----------')
    f1_file = open(f1_dir, 'a')
    f1_file.write(f'[Time: {time.asctime()}] F1 of {dataset}_{method}_{backbone} is : {f1}\n')
    print(f'Macro-F1 result: {f1} have been saved to {f1_dir}.')

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='analyse result')
    parser.add_argument('--dataset', type=str, default='KLSG')
    parser.add_argument('--method', type=str, default='betl')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--get_gmean', type=str, default='True', help='If you want to get Gmean result, default is True.')
    parser.add_argument('--get_f1', type=str, default='True', help='If you want to get Macro-F1 result, default is True.')
    parser.add_argument('--get_conf_matrix', type=str, default='False', help='If you want to get confusion matrix result, default is False.')
    parser.add_argument('--get_pr', type=str, default='False', help='If you want to get Precision-Recall curves result, default is False.')
    parser.add_argument('--get_macro_pr_all', type=str, default='False', help='If you want to get macro average Precision-Recall curves result of all classes, default is False.')
    parser.add_argument('--get_macro_pr_tail', type=str, default='False', help='If you want to get macro average Precision-Recall curves result of tail classes, default is False.')
    parser.add_argument('--show_pic', type=str, default='False', help='If you want to show confusion matrix or Precision-Recall curves, default is False.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get args
    args = parse_args()

    # set params
    dataset = args.dataset
    method = args.method
    backbone = args.backbone
    plt.rc('font', family='Times New Roman', size=19) # size=19
    curr_dir = os.path.dirname(__file__)
    y_true, y_hat, logits = get_y_and_logits_results(dataset, backbone, method, if_get_logits=True)
    if dataset == 'KLSG':
        num_class = 2
        class_list = ['Plane', 'Wreck']
        tail_idxes = [0]
        colors = ['orangered', 'lightseagreen']
    elif dataset == 'LTSID':
        num_class = 8
        class_list = ['C_Seabed', 'D_Victim', 'Plane', 'G_Seabed', 
                    'S_Seabed', 'Tire', 'Valve', 'Wreck']
        tail_idxes = [0, 1, 2, 3, 6]
        colors = ['orangered', 'lightpink', 'coral', 'deeppink', 
                  'lightseagreen', 'steelblue', 'magenta', 'cyan']
    elif dataset == 'FLSMDD':
        num_class = 10
        class_list = ['Bottle', 'Can', 'Chain', 'D_Carton', 'Hook',
                    'Propeller', 'Sh_Bottle', 'St_Bottle', 'Tire', 'Valve']
        tail_idxes = [2, 4, 5, 6, 7, 9]
        colors = ['lightseagreen', 'steelblue', 'deeppink', 'blue', 'orangered', 
                  'lightpink', 'coral', 'magenta', 'cyan', 'red']
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')

    # write gmean
    if args.get_gmean in ['True', 'true']:
        gmean = cal_gmean(y_true, y_hat)
        gmean_dir = os.path.join(curr_dir, f'../output/display/gmean.txt') 
        write_gmean(gmean, gmean_dir, dataset, method, backbone)
        # gmean_list = get_gmean_each_trial(dataset, backbone, method)
        # print(f'gmean list is : {gmean_list}')

    # write f1
    if args.get_f1 in ['True', 'true']:
        f1 = f1_score(y_true, y_hat, average='macro')
        f1_dir = os.path.join(curr_dir, f'../output/display/f1.txt') 
        write_f1(f1, f1_dir, dataset, method, backbone)

    # save confusion matrix
    if args.get_conf_matrix in ['True', 'true']:
        save_folder = os.path.join(curr_dir, f'../output/display/cm')
        cm_dir = os.path.join(save_folder, f'{dataset}_{method}_{backbone}.pdf') 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        draw_conf_matrix(class_list, y_true, y_hat, tail_idxes, cm_dir)

    # save pr curve
    if args.get_pr in ['True', 'true']:
        save_folder = os.path.join(curr_dir, f'../output/display/pr')
        pr_dir = os.path.join(save_folder, f'{dataset}_{method}_{backbone}.pdf')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if 'svm' in method:
            probs = logits
        else:
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        draw_pr_curve(y_true, probs, class_list, colors, pr_dir)

    # save macro-pr curve
    if args.get_macro_pr_all in ['True', 'true']:
        save_folder = os.path.join(curr_dir, f'../output/display/macro_pr')
        macro_pr_dir = os.path.join(save_folder, f'all_classes_{dataset}_{backbone}.pdf')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        draw_macro_pr_curve(dataset, num_class, macro_pr_dir)

    # save tail-classes macro-pr curve
    if args.get_macro_pr_tail in ['True', 'true']:
        save_folder = os.path.join(curr_dir, f'../output/display/macro_pr')
        tail_macro_pr_dir = os.path.join(save_folder, f'tail_classes_{dataset}_{backbone}.pdf')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        draw_tail_classes_pr_curve(dataset, num_class, tail_idxes, tail_macro_pr_dir)
    
    if args.show_pic in ['True', 'true']:
        plt.show()