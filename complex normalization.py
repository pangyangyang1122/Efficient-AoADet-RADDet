import os
import numpy as np
from tqdm import tqdm
from glob import glob

import util.helper as helper

"""
计算复数的模长的均值和方差
"""

def calc_mean_std(data_dir, outfilename):

    RD_files = glob(os.path.join(data_dir, '*', '*.npy'))

    # for f in tqdm(RAD_files, total=len(RAD_files)):

    global_mean_log = 0
    global_std_log  = 0
    max_mag  = []
    min_mag  = []

    datanum = len(RD_files)
    # datanum = 10
    for i in tqdm(range(datanum)):
        filename = RD_files[i]
        RD_complex = np.load(filename)

        RD_mag = np.abs(RD_complex)


        max_mag.append(RD_mag.max())
        min_mag.append(RD_mag.min())


    global_max_mag = max(max_mag)
    global_min_mag = min(min_mag)


    fid = open(outfilename, 'w')
    fid.write(outfilename+"\n")

    fid.write(f"global max  magnitude: {global_max_mag}\n")
    fid.write(f"global min  magnitude: {global_min_mag}\n\n")
    fid.close()

if __name__ == "__main__":

    # author_RAD_dir = '/media/ljm/disk2/RADDet_DATASET/train/RAD/'
    # calc_mean_std(author_RAD_dir, 'RADDet_statistics_author.txt')

    # my_RAD_dir = '/media/ljm/disk2/RADDet_reprocessed/train/RAD/'
    # calc_mean_std(my_RAD_dir, 'RADDet_statistics_RAD.txt')

    # my_RAD4_dir = '/media/ljm/disk2/RADDet_reprocessed/train/RAD4/'
    # calc_mean_std(my_RAD4_dir, 'RADDet_statistics_RAD4.txt')

    RD_dir = '/media/sqpang/新加卷/rada_target_detection/Data/RADDet_reprocessed/test/RAD'
    calc_mean_std(RD_dir, 'RAD_complex_mod_mean_std.txt')
