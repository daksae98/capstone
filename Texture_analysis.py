import numpy as np
import pickle
import matplotlib.pyplot as plt

'''
for index,matchInfo in enumerate(pkl):
    kp1_count=matchInfo['kp1_count'],
    kp2_count=matchInfo['kp2_count'],
    matches_count=matchInfo['matches_count'],
    mean_contrast=matchInfo['mean_contrast'],
    mean_dissimilarity=matchInfo['mean_dissimilarity'],
    mean_homogeneity=matchInfo['mean_homogeneity'],
    mean_energy=matchInfo['mean_energy'],
    mean_correlation=matchInfo['mean_correlation'],
    mean_ASM=matchInfo['mean_ASM'],
    inliers_count=matchInfo['inliers_count']
    
    print(f'{index}/{len(pkl)} : mean_contrast : {(matchInfo)}')
'''
PKL_PATH = 'pkls/merged/'
ALGO_TYPES = ['KAZE','ORB','SIFT']
PLOT_COLORS = {
    'KAZE' : '#6573c2',
    'ORB' : '#916e70',
    'SIFT' : '#91b89b',
}
LINE_COLORS = {
    'KAZE' : '#0018f0',
    'ORB' : '#d60909',
    'SIFT' : '#00c732',
}
pkls = {}
for algo in ALGO_TYPES:
    with open(f'{PKL_PATH+algo}.pkl', 'rb') as f:
        pkl = pickle.load(f)
        pkls[algo] = pkl





# def plot_scatter(xlabel,ylabel):
#     plt.plot([matchInfo[xlabel] for matchInfo in sorted_pkl], [matchInfo[ylabel] for matchInfo in sorted_pkl],marker='o',ls='-',ms=4, label='SIFT')  # Plot some data on the (implicit) axes.
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(xlabel.upper())
#     plt.legend()
#     plt.show()    



def plot_algo_xlabel_ylabel(xlabel,ylabel):
    sorted_pkls = {}
    for algo in ALGO_TYPES:
        pkl = pkls[algo]
        sorted_pkls[algo] = sorted(pkl, key=lambda x: x[xlabel])
        x = np.array([matchInfo[xlabel] for matchInfo in sorted_pkls[algo]])
        y = np.array([matchInfo[ylabel] for matchInfo in sorted_pkls[algo]])
        plt.plot(x,y ,marker='o',ls='-', color=PLOT_COLORS[algo],label=algo)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x+b,color=LINE_COLORS[algo],linewidth=3,label=f'{algo} regression line')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show() 

def subplot_algo_xlabel_ylabel(lable_list):
    length = len(lable_list)
    fig, axs = plt.subplots(length,1)
    
    for index,(xlabel,ylabel) in enumerate(lable_list):
        sorted_pkls = {}
        for algo in ALGO_TYPES:
            pkl = pkls[algo]
            sorted_pkls[algo] = sorted(pkl, key=lambda x: x[xlabel])
            x = np.array([matchInfo[xlabel] for matchInfo in sorted_pkls[algo]])
            y = np.array([matchInfo[ylabel] for matchInfo in sorted_pkls[algo]])
            axs[index].plot(x,y ,marker='o',ls='-', color=PLOT_COLORS[algo],label=algo)
            m, b = np.polyfit(x, y, 1)
            axs[index].plot(x, m*x+b,color=LINE_COLORS[algo],linewidth=3,label=f'{algo} regression line')
        axs[index].set_xlabel(xlabel)
        axs[index].set_ylabel(ylabel)
        axs[index].legend()
    plt.show()   


def plot_algo_xlabel_ratio(xlabel):
    sorted_pkls = {}
    for algo in ALGO_TYPES:
        pkl = pkls[algo]
        sorted_pkls[algo] = sorted(pkl, key=lambda x: x[xlabel])
        x = np.array([matchInfo[xlabel] for matchInfo in sorted_pkls[algo]])
        y = np.array([matchInfo['inliers_count']/matchInfo['matches_count'] for matchInfo in sorted_pkls[algo]])
        plt.plot(x,y,marker='o',ls='-', label=algo)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x+b,color=LINE_COLORS[algo],linewidth=3,label=f'{algo} regression line')
    plt.xlabel(xlabel)
    plt.ylabel('inlier ratio')
    plt.legend()
    plt.show()  

def subplot_algo_xlabel_ratio(xlabel_list):
    length = len(xlabel_list)
    fig, axs = plt.subplots(length,1)
    
    for index,xlabel in enumerate(xlabel_list):
        sorted_pkls = {}
        for algo in ALGO_TYPES:
            pkl = pkls[algo]
            sorted_pkls[algo] = sorted(pkl, key=lambda x: x[xlabel])
            x = np.array([matchInfo[xlabel] for matchInfo in sorted_pkls[algo]])
            y = np.array([matchInfo['inliers_count']/matchInfo['matches_count'] for matchInfo in sorted_pkls[algo]])
            axs[index].plot(x,y,marker='o',ls='-', label=algo)
            m, b = np.polyfit(x, y, 1)
            axs[index].plot(x, m*x+b,color=LINE_COLORS[algo],linewidth=3,label=f'{algo} regression line')
        axs[index].set_xlabel(xlabel)
        axs[index].set_ylabel('inlier ratio')
        axs[index].legend()
    fig.suptitle('Inlier Ratio')
    plt.show()  



def analysis_images():
    # 영상 별 평균 GLCM 통계값
    stats = [
        'mean_contrast',
        'mean_dissimilarity',
        'mean_homogeneity',
        'mean_energy',
        'mean_correlation',
        'mean_ASM',
    ]
    matchInfos = pkls['KAZE']

    def cal_min_max_mean(arr):
        return [np.min(arr), np.max(arr), np.mean(arr)]
    pth = PKL_PATH.split('/')[1]
    with open(f'image_analysis/{pth}.txt','w+') as f:
        f.write(f'{PKL_PATH}\n')
        for stat in stats:
            result = cal_min_max_mean([x[stat] for x in matchInfos])
            f.write(f'{stat} : min : {result[0]}, max : {result[1]}, mean : {result[2]}\n')
        f.close()

if __name__ == '__main__':
    lable_list = [
    ('mean_contrast','kp2_count'),
    ('mean_energy','kp1_count'),
    # ('mean_correlation','inliers_count'),
    # ('mean_ASM','inliers_count'),
    # ('mean_dissimilarity','inliers_count'),
    # ('mean_homogeneity','inliers_count'),
    ]
    xlable_list = [
        'mean_contrast',
        'mean_energy',
        'mean_correlation',
        'mean_ASM',
    ]
    # subplot_algo_xlabel_ratio(xlable_list)
    # subplot_algo_xlabel_ylabel(lable_list)
    # analysis_images()
    plot_algo_xlabel_ylabel('mean_contrast','inliers_count')
    # plot_algo_xlabel_ratio('mean_contrast')