import os
import pickle
ALGO_TYPES = ['KAZE','ORB','SIFT']
MERGE_PKL_PATH = 'pkls/merged/'

def merge_pkls():
    try:
        if not os.path.exists(MERGE_PKL_PATH):
                os.makedirs(MERGE_PKL_PATH)
    except OSError:
        print ('Error: Creating directory. ' +  MERGE_PKL_PATH)
    dir_list = os.listdir('pkls')
    merge_pkl = {
        'KAZE' :[],
        'ORB' : [],
        'SIFT' : [],
    }

    for directory in dir_list:
        for algo in ALGO_TYPES:
            if directory == 'match-order':
                 continue
            if directory.startswith('.'):
                 continue
            if f'pkls/{directory}/' == MERGE_PKL_PATH:
                    continue
            with open(f'pkls/{directory}/{algo}.pkl','rb') as f:
                pkl = pickle.load(f)
                merge_pkl[algo] = merge_pkl[algo] + pkl
                print(len(pkl))

    print(len(merge_pkl['KAZE']))
    for algo in ALGO_TYPES:
        with open(f'{MERGE_PKL_PATH+algo}.pkl', 'wb') as f:
            pickle.dump(merge_pkl[algo], f)

if __name__ == '__main__':
     merge_pkls()