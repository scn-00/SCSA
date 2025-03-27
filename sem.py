import numpy as np
from sklearn.cluster import KMeans
import cv2
import numpy as np
import torch
import os


red = [0, 0, 230]
cyan = [230, 230, 0]
pink = [230, 0, 230]
blue = [230, 0, 0]
green = [0, 230, 0]
yellow = [0, 230, 230]
dark_red = [0, 0, 115]
grey = [230, 230, 230]
fleshcolor = [163, 225, 227]



labels = [
    ([1], np.array([[0, 0, 255], [0, 255, 255], [64, 192, 0], [0, 255, 0], [191, 191, 191], [192, 255, 192], [255, 0, 0], [64, 192, 64]])),
    ([2], np.array([[0, 0, 255], [0, 255, 255], [0, 255, 192], [0, 255, 0], [191, 191, 191], [255, 0, 0]])),
    ([3, 5, 6, 11, 16, 19, 21, 24, 25, 28, 29, 32, 37, 40, 44, 45, 48, 54, 56, 60, 66], np.array([green, yellow, red, blue])),
    ([4], np.array([blue, green, yellow])),
    ([7, 12, 13, 18, 23, 26, 27, 30, 31, 34, 36, 38, 41, 46, 53, 55, 57, 58], np.array([yellow, green, red])),
    ([8, 33, 35, 47, 51, 52, 61, 62, 63, 64, 68], np.array([yellow, green, red, blue, pink])),
    ([9, 10, 42, 43, 69], np.array([green, red])),
    ([14], np.array([yellow, green, red, blue, grey])),
    ([15], np.array([yellow, green, red, blue, grey, cyan, pink])),
    ([17, 22, 50], np.array([yellow, green, red, blue, grey, cyan, dark_red, pink])),
    ([20], np.array([yellow, green, red, blue, grey, cyan, fleshcolor, pink, dark_red])),
    ([39, 49, 59, 65, 67], np.array([yellow, green, red, blue, pink, cyan]))
]

# 创建颜色映射字典
labels_color = {}
for numbers, colors in labels:
    for num in numbers:
        labels_color[num] = colors






def read_colorful_mask(path, content_path_sem, style_path_sem, size):

    c_name = "_".join(content_path_sem.split('/')[-1].split('_')[:-1])
    s_name = "_".join(style_path_sem.split('/')[-1].split('_')[:-1])
    map_path = c_name + '_' + s_name + '_' + 'map_'
    number = content_path_sem.split('/')[-2]
    feat_path = path + str(number)
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

    content_mask = cv2.imread(content_path_sem)
    style_mask = cv2.imread(style_path_sem)
    
    content_mask = cv2.resize(content_mask, (512, 512), cv2.INTER_NEAREST) 
    style_mask = cv2.resize(style_mask, (512, 512), cv2.INTER_NEAREST)
    
    content_mask = cv2.resize(content_mask, (size, size), cv2.INTER_NEAREST) 
    style_mask = cv2.resize(style_mask, (size, size), cv2.INTER_NEAREST)

    
    # flatten
    content_shape = content_mask.shape[0:2]
    content_mask = content_mask.reshape([content_shape[0]*content_shape[1], -1])
    style_shape = style_mask.shape[0:2]
    style_mask = style_mask.reshape([style_shape[0]*style_shape[1], -1])
    
    label = labels_color.get(int(number))
    
    n_colors = label.shape[0]
    # cluster
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(label)

    # predict
    content_labels = kmeans.predict(content_mask)
    content_label = content_labels
    content_labels = content_labels.reshape([content_shape[0], content_shape[1]])
    style_labels = kmeans.predict(style_mask)
    style_label = style_labels
    style_labels = style_labels.reshape([style_shape[0], style_shape[1]])
    
    # stack
    content_masks = []
    style_masks = []
    for i in range(n_colors):
        content_masks.append( (content_labels == i).astype(np.float32) )
        style_masks.append( (style_labels == i).astype(np.float32) )
        
    c_sem = torch.from_numpy(np.stack(content_masks))
    s_sem = torch.from_numpy(np.stack(style_masks))
    torch.save(c_sem, path  +'/' + number + '/' + c_name + '_masks_' + str(size) + ".pt")
    torch.save(s_sem, path  +'/' + number + '/' + s_name + '_masks_' + str(size) + ".pt")    
        
    if size < 256:
        # map
        map = np.zeros((size*size,size*size))
        for i in range(size*size):
            for j in range(size*size):
                if content_label[i] == style_label[j]:
                    map[i,j] = 1
        map =  torch.tensor(map)
        torch.save(map, path  +'/' + number + '/' + map_path + str(size) + ".pt")

        return np.stack(content_masks), np.stack(style_masks), map, number, n_colors
    
    else:
        return np.stack(content_masks), np.stack(style_masks), None, number, n_colors
    


def get_sem_mask(path, c_sem_path, s_sem_path): 
    content_masks, style_masks, sem_map, number, n_colors = read_colorful_mask(path, c_sem_path, s_sem_path, 32)
    content_masks, style_masks, sem_map, number, n_colors = read_colorful_mask(path, c_sem_path, s_sem_path, 64)
    content_masks, style_masks, sem_map, number, n_colors = read_colorful_mask(path, c_sem_path, s_sem_path, 128)
 

    content_masks, style_masks, sem_map, number, n_colors = read_colorful_mask(path, s_sem_path, c_sem_path, 32)
    content_masks, style_masks, sem_map, number, n_colors = read_colorful_mask(path, s_sem_path, c_sem_path, 64)
    content_masks, style_masks, sem_map, number, n_colors = read_colorful_mask(path, s_sem_path, c_sem_path, 128)
    dir_path = path + str(number) + '/fig'
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    
    for i in range(n_colors):
        cv2.imwrite(dir_path+ '/c_'+str(i)+'.png', content_masks[i]*255)
        cv2.imwrite(dir_path+ '/s_'+str(i)+'.png', style_masks[i]*255)
    cv2.imwrite(dir_path+ '/map.png', sem_map.numpy()*255)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start',type=int, default = 0)
parser.add_argument('--phase',type=int, default = 0)
parser.add_argument('--data_path', default = 'sem_data')

opt = parser.parse_args()

sem_data_list = []
data_folders = [f.name for f in os.scandir(opt.data_path) if f.is_dir()]
for data_folder in data_folders:
    if data_folder.isdigit():
        files = [f.name for f in os.scandir(opt.data_path + '/' + data_folder) if f.is_file()]
        data_list = []
        for file in files:
            data_list.append(opt.data_path + '/' + data_folder + '/' + file)
        sem_data_list.append(data_list)
sem_data_list.sort()   

if opt.phase == 0:
    opt.phase = len(sem_data_list)


def get_data(data_list):
    data_dict = {}
    for data in data_list:
        if '_sem' in data:
            if 'cnt_sem' not in data_dict.keys():
                data_dict['cnt_sem'] = data
                data_dict['cnt'] = data.replace("_sem.png",".jpg")
            else:
                data_dict['sty_sem'] = data
                data_dict['sty'] = data.replace("_sem.png",".jpg")
    return data_dict['cnt'], data_dict['sty'], data_dict['cnt_sem'], data_dict['sty_sem']
     
                
for i in range(opt.start, opt.start + opt.phase):
    
    cnt, sty, cnt_sem, sty_sem = get_data(sem_data_list[i])
   
    folder_name = cnt.split('/')[-2]
    get_sem_mask('sem_precomputed_feats/', 
                cnt_sem,
                sty_sem)
