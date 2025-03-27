import subprocess
import os



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cnt', default = '../../sem_data/1/1.jpg')
parser.add_argument('--sty', default = '../../sem_data/1/1_paint.jpg')
parser.add_argument('--cnt_sem', default = '../../sem_data/1/1_sem.png')
parser.add_argument('--sty_sem', default = '../../sem_data/1/1_paint_sem.png')
parser.add_argument('--sem_map_64', default = '../../sem_precomputed_feats/1/1_1_paint_map_64.pt')
parser.add_argument('--sem_map_32', default = '../../sem_precomputed_feats/1/1_1_paint_map_32.pt')
opt = parser.parse_args()

cnt = opt.cnt
sty = opt.sty
cnt_sem = opt.cnt_sem
sty_sem = opt.sty_sem
sem_map_64 = opt.sem_map_64
sem_map_32 = opt.sem_map_32

os.chdir('../StyleID')
result = subprocess.run(['python', 'get_pre_features.py', '--cnt', cnt, '--sty', sty, '--cnt_sem', cnt_sem, '--sty_sem', opt.sty_sem], capture_output=True,text=True)

os.chdir('../StyleID+SCSA')

folder_name = cnt.split('/')[-2]

result = subprocess.run(['python', 'SCSA.py', '--cnt', cnt, '--sty', sty, '--cnt_sem', cnt_sem, '--sty_sem', sty_sem,
                '--sem_map_64', sem_map_64, '--sem_map_32', sem_map_32],
            capture_output=True,text=True)

print(result.stderr)











