import subprocess
import os
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default = '../sem_data')
opt = parser.parse_args()

sem_data_list = []
data_folders = [f.name for f in os.scandir(opt.data_path) if f.is_dir()]
data_folders = [item for item in data_folders if item.isdigit()]
data_folders = sorted(data_folders, key=lambda x: int(x))


for data_folder in data_folders:
    files = [f.name for f in os.scandir(opt.data_path + '/' + data_folder) if f.is_file()]
    data_list = []
    for file in files:
        data_list.append(opt.data_path + '/' + data_folder + '/' + file)
    sem_data_list.append(data_list)
    
    
def get_data(data_list):
    data_dict = {}
    for data in data_list:
        if '_sem' in data:
            if 'cnt_sem' not in data_dict.keys():
                data_dict['cnt_sem'] = data
            else:
                data_dict['sty_sem'] = data

    for data in data_list:
        if '_sem' in data:
            continue
        else:
            name = data.split('/')[-1].split('.')[0]
            for k, v in data_dict.items():
                if v.split('/')[-1].split('.')[0].replace('_sem','') ==name:
                    data_dict[k.replace('_sem','')] = data
                    break

    return data_dict['cnt'], data_dict['sty'], data_dict['cnt_sem'], data_dict['sty_sem']



for i in range(0, len(data_folders)):
    start = i
    cnt, sty, cnt_sem, sty_sem = get_data(sem_data_list[i])
    folder_name = cnt.split('/')[1]
    c_name = cnt.split('/')[-1].split('.')[0]
    s_name = sty.split('/')[-1].split('.')[0]
    number = sty.split('/')[-2]
    sem_map_32 = '../sem_precomputed_feats/' + number + '/' + c_name + '_' + s_name + '_map_32.pt'
    sem_map_64 = '../sem_precomputed_feats/' + number + '/' + c_name + '_' + s_name + '_map_64.pt'
    result = subprocess.run(['python', 'SCSA.py', '--content', cnt, '--style', sty, '--content_sem', cnt_sem, '--style_sem', sty_sem,
                    '--sem_map_64', sem_map_64, '--sem_map_32', sem_map_32],
                capture_output=True, text=True)



    c_name = sty.split('/')[-1].split('.')[0]
    s_name = cnt.split('/')[-1].split('.')[0]
    sem_map_32 = '../sem_precomputed_feats/' + number + '/' + c_name + '_' + s_name + '_map_32.pt'
    sem_map_64 = '../sem_precomputed_feats/' + number + '/' + c_name + '_' + s_name + '_map_64.pt'
    result = subprocess.run(['python', 'SCSA.py', '--content', sty, '--style', cnt, '--content_sem', sty_sem, '--style_sem', cnt_sem,
                    '--sem_map_64', sem_map_64, '--sem_map_32', sem_map_32],
                capture_output=True, text=True)





    cnt, sty, cnt_sem, sty_sem = get_data(sem_data_list[i])

    result = subprocess.run(['python', 'SANet.py', '--content', cnt, '--style', sty],
                capture_output=True,text=True)


    result = subprocess.run(['python', 'SANet.py', '--content', sty, '--style', cnt],
                capture_output=True,text=True)

