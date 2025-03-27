import subprocess
import os


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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default = '../sem_data')
opt = parser.parse_args()

sem_data_list = []
data_folders = [f.name for f in os.scandir(opt.data_path) if f.is_dir()]
data_folders = [item for item in data_folders if item.isdigit()]
data_folders = sorted(data_folders, key=lambda x: int(x))


for data_folder in data_folders:
    if data_folder.isdigit():
        files = [f.name for f in os.scandir(opt.data_path + '/' + data_folder) if f.is_file()]
        data_list = []
        for file in files:
            data_list.append(opt.data_path + '/' + data_folder + '/' + file)
        sem_data_list.append(data_list)




start = 0

for i in range(0, len(data_folders)): 
    

    cnt, sty, cnt_sem, sty_sem = get_data(sem_data_list[i])
    
    os.chdir('StyleID')
    r = subprocess.run(['python', 'get_pre_features.py', '--cnt', '../' + cnt, '--sty', '../' + sty, '--cnt_sem', '../' + cnt_sem, '--sty_sem', '../' + sty_sem], capture_output=True,text=True)
    

    os.chdir('..')
    os.chdir('StyleID+SCSA')

    folder_name = cnt.split('/')[-2]
    c_name = cnt.split('/')[-1].split('.')[0]
    s_name = sty.split('/')[-1].split('.')[0]
    sem_map_32 = '../../sem_precomputed_feats/' + folder_name + '/' + c_name + '_' + s_name + '_map_32.pt'
    sem_map_64 = '../../sem_precomputed_feats/' + folder_name + '/' + c_name + '_' + s_name + '_map_64.pt'
    result = subprocess.run(['python', 'SCSA.py', '--cnt', '../' + cnt, '--sty', '../' + sty, '--cnt_sem', '../' + cnt_sem, '--sty_sem', '../' + sty_sem,
                    '--sem_map_64', sem_map_64, '--sem_map_32', sem_map_32],
                capture_output=True,text=True)


    c_name = cnt.split('/')[-1].split('.')[0]
    s_name = sty.split('/')[-1].split('.')[0]
    sem_map_32 = '../../sem_precomputed_feats/' + folder_name + '/' + s_name + '_' + c_name + '_map_32.pt'
    sem_map_64 = '../../sem_precomputed_feats/' + folder_name + '/' + s_name + '_' + c_name + '_map_64.pt'
    result = subprocess.run(['python', 'SCSA.py', '--cnt', '../' + sty, '--sty','../' + cnt, '--cnt_sem', '../' + sty_sem, '--sty_sem', '../' + cnt_sem,
                    '--sem_map_64', sem_map_64, '--sem_map_32', sem_map_32],
                capture_output=True,text=True)


    cnt, sty, cnt_sem, sty_sem = get_data(sem_data_list[i])
    os.chdir('..')
    os.chdir('StyleID')
    subprocess.run(['python', 'StyleID.py', '--cnt', '../' + cnt, '--sty', '../' + sty], capture_output=True,text=True)
    subprocess.run(['python', 'StyleID.py', '--cnt', '../' + sty, '--sty', '../' + cnt], capture_output=True,text=True)   
    os.chdir('..')






