#!/usr/bin/env python

import sys, os


encode_list=sys.argv[1:] # doc2vec net go
divide=['c1','c3']

for i in range(1,4):
    for d in divide:
        filedir='1_10_'+d+'_'+str(i)
        newdir='../sample/'+filedir+'/'+''.join(encode_list)+'/'+filedir+'/'
        os.system('mkdir -p '+newdir)
        for j in ['set1','set2','set3','set4','set5','independent_test']:
            w=open(newdir+j, 'w')
            ppi_encodes={}
            for encode in encode_list:
                name='../sample/'+filedir+'/'+encode+'/'+filedir+'/'
                r=open(name+j)
                for line in r.readlines():
                    line=line.strip()
                    before=' '.join(line.split()[0:3])
                    feature=line.split()[3:]
                    if before not in ppi_encodes:
                        ppi_encodes[before]=feature
                    else:
                        ppi_encodes[before]+=feature
                r.close()
            for each in ppi_encodes:
                w.write(each+' '+' '.join(ppi_encodes[each])+'\n')
            w.close()


