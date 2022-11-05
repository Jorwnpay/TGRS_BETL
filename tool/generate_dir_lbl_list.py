# coding:utf-8
# This code is based on https://github.com/TingsongYu/PyTorch_Tutorial/blob/master/Code/1_data_prepare/1_3_generate_txt.py.
import os
import argparse

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('--dataset', type=str, default='KLSG')
    args = parser.parse_args()
    return args

def gen_dir_lbl_list(save_path, img_dir):
    label_idx = -1
    f = open(save_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  
        sorted_s_dirs = [sub_dir for sub_dir in s_dirs] # sort the folders by folder name
        sorted_s_dirs.sort()
        for sub_dir in sorted_s_dirs:
            label_idx += 1
            i_dir = os.path.join(root, sub_dir)   
            img_list = os.listdir(i_dir)                    
            for i in range(len(img_list)):
                if not (img_list[i].endswith('png') or img_list[i].endswith('jpg')):         
                    continue
                label = label_idx
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + str(label) + '\n'
                f.write(line)
    f.close()

if __name__ == '__main__':
    args = parse_args()
    assert args.dataset in ['KLSG', 'FLSMDD']
    curr_dir = os.path.dirname(__file__)
    save_path = os.path.join(curr_dir, '../data', f'{args.dataset}', 'train.txt')
    img_dir = os.path.join(curr_dir, '../data', f'{args.dataset}')
    gen_dir_lbl_list(save_path, img_dir)
    print(f'The generated file has been saved to {save_path}.')
