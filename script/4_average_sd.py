#!/usr/bin/env python
import numpy as np
import os

def average_sd(target_list):
    average='%.3f'%np.average(target_list)
    sd='%.3f'%np.std(target_list)
    return(average+'+-'+sd)

def get_target_list(encode, filedir, origidir, algorithm):
    cross_auc, cross_auprc, independent_auc, independent_auprc=[], [], [], []
    for i in range(1,4):
        filename=origidir+filedir+str(i)+'/'+encode+'/'+filedir+str(i)+'/'+algorithm+'/parameters'
        print(filename)
        independent_auc.append(float(os.popen('grep independent_test '+filename).read().strip().split('\t')[1]))
        independent_auprc.append(float(os.popen('grep independent_test '+filename).read().strip().split('\t')[2]))
        cross_auc.append(float(os.popen("grep 'Average AUC' "+filename).read().strip('\n').split(': ')[1]))
        cross_auprc.append(float(os.popen("grep 'Average APRUC' "+filename).read().strip('\n').split(': ')[1]))
    cross_auc, cross_auprc, independent_auc, independent_auprc=average_sd(cross_auc), average_sd(cross_auprc), average_sd(independent_auc), average_sd(independent_auprc)
    return(cross_auc, cross_auprc, independent_auc, independent_auprc)

def main():
    import sys

    origidir='sample/'
    filedirold='1_10_'
    encodes=sys.argv[1].split() # ['doc2vec_cdhit13256702']
    divide=sys.argv[2].split() #['c1','c3']
    algorithms=sys.argv[3].split() #['lgbm']

    w=open('../Run_result.txt', 'a')
    w.write(filedirold+'\n')
    for di in divide:
        for encode in encodes:
            for algorithm in algorithms:
                filedir=filedirold+di+'_'
                cross_auc, cross_auprc, independent_auc, independent_auprc=get_target_list(encode, filedir, origidir, algorithm)
                w.write(di+'\t'+encode+'\t'+algorithm+'\t'+'\t'.join([cross_auc, cross_auprc, independent_auc, independent_auprc])+'\n')
    w.close()
        

if __name__=='__main__':
    main()
