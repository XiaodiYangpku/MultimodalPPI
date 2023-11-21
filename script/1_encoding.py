#!/usr/bin/env python
#-*-coding:utf-8 -*-
from __future__ import division
import re, pickle
import os
import numpy as np
import pandas as pd
from os.path import join
from collections import Counter
from collections import defaultdict
from Bio import SeqIO
from joblib import Parallel
from joblib import delayed
import cython
import multiprocessing
from multiprocessing.pool import Pool


def get_documents(seq_list, seq_ids, start, end, k, extract_method):
    """
    :param seq_list:
    :param seq_ids:
    :param start:
    :param end:
    :param k:
    :param extract_method
    Example sequence: MALFFFNNN
    doc2vec parameters: k, extract_method, vector_size, window
    extract_method: [1, 2, 3]
        sequence example: MALFFFNNN
        1) ['MAL', 'ALF', 'LFF', 'FFF', 'FFN', 'FNN', 'NNN']
        2) ['MAL', 'FFF', 'NNN', 'ALF', 'FFN', 'LFF', 'FNN']
        3) ['MAL', 'FFF', 'NNN']
           ['ALF', 'FFN']
           ['LFF', 'FNN']
    vector_size: [64]
    window [3]
    :return:
    """
    from gensim.models.doc2vec import TaggedDocument
    documents = []

    for seq, seq_id in zip(seq_list, seq_ids):
        codes = seq[start: end]
        if extract_method == 4:
            words1 = [codes[i: i + k] for i in range(len(codes) - (k - 1))]
            words2= [codes[j: j + k] for i in range(k) for j in range(i, len(codes) - (k - 1), k)]
            words=words1+words2
            documents.append(TaggedDocument(words, tags=[seq]))
        elif extract_method == 1:
            words = [codes[i: i + k] for i in range(len(codes) - (k - 1))]
            documents.append(TaggedDocument(words, tags=[seq]))
        elif extract_method == 2:
            words = [codes[j: j + k] for i in range(k) for j in range(i, len(codes) - (k - 1), k)]
            documents.append(TaggedDocument(words, tags=[seq]))
        elif extract_method == 3:
            """
            # output:
                ['MAL', 'FFF', 'NNN']
                ['ALF', 'FFN']
                ['LFF', 'FNN']
            """
            for i in range(k):
                words = [codes[j: j + k] for j in range(i, len(codes) - (k - 1), k)]
                documents.append(TaggedDocument(words, tags=[seq + '_%s' % i]))
    return documents


def get_doc2vec_parameters(encode):
    """
    k, extract_method, vector_size, window, epoch
    :param encode:
    :return:
    """
    elements = [int(i) for i in encode.split('-')[2:]]
    try:
        k, extract_method, vector_size, window, epoch = elements
        return k, extract_method, vector_size, window, epoch
    except Exception as e:
        print('Error, the number of hyper-parameter of doc2vec is unequal!!!', elements)
        exit()


def doc2vector(seq_list, seq_ids, start, end, k, extract_method, vector_size, window, epoch):
    """
    :param seq_list:
    :param seq_ids:
    :param start:
    :param end:
    :return: feature_matrix, feature name
    """
    from gensim.models.doc2vec import Doc2Vec
    feature_matrix = []  
    documents = get_documents(seq_list, seq_ids, start, end, k, extract_method)

    print('doc2vec sequence length', len(documents))
    # https://radimrehurek.com/gensim/models/doc2vec.html

    model = Doc2Vec(documents, min_count=1, window=window, vector_size=vector_size, sample=1e-3, negative=5, workers=40)
    #xdyang:min_count (int, optional) – Ignores all words with total frequency lower than this
    #xdyang:window (int, optional) – The maximum distance between the current and predicted word within a sentence.
    #xdyang:vector_size (int, optional) – Dimensionality of the feature vectors.
    #xdyang:sample (float, optional) – The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    #xdyang:alpha (float, optional) – The initial learning rate.
    #xdyang:negative (int, optional) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    #xdyang:workers (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines).
    print('start train doc2vec model')
    model.train(documents, total_examples=model.corpus_count, epochs=epoch)
    print('doc2vec model training finished')
    for index, seq in enumerate(seq_list):
        if extract_method == 3:
            # contain k sub-sequences
            feature_matrix.append([np.mean([model[seq + '_%s' % j][i] for j in range(k)]) for i in range(vector_size)])
        else:
            feature_matrix.append([model[seq][i] for i in range(vector_size)])

    return feature_matrix, ['vector size: ' + str(vector_size)], model


def infer_vector(seq_ids, seqs, start, end, encode, model_fname, encoding_fname, complete_infer):
    # get doc2vec parameters
    from gensim.models.doc2vec import Doc2Vec
    k, extract_method, vector_size, window, epoch = get_doc2vec_parameters(encode)
    documents = get_documents(seqs, seq_ids, start, end, k, extract_method)
    model = Doc2Vec.load(model_fname)
    protein_encodings = pickle.load(open(encoding_fname, 'rb'))

    feature_matrix = []
    infernum=0
    unifernum=0
    if complete_infer:
        print('All proteins are inferred!!!')
    for seq_id, seq, document in zip(seq_ids, seqs, documents):
        if complete_infer:
            # complete inferring by doc2vec model
            # print(model.infer_vector(document, alpha=0.001, steps=epoch))
            feature_matrix.append(list(model.infer_vector(document[0])))
        else:
            # if seq_in in protein_encodings protein_encodings[seq_id] else inferring by doc2vec model
            if seq in protein_encodings:
                feature_matrix.append(protein_encodings[seq])
                unifernum+=1
            else:
                print(seq_id, 'is inferred!!!')
                infernum+=1
                feature_matrix.append(list(model.infer_vector(document[0])))
    print('Infered protein number:',infernum)
    print('Uninfered protein number:',unifernum)
    return dict(zip(seqs, feature_matrix))


def extract_seq(fasta_fname, min_len):
    seq_ids, seqs = [], []
    for seq_record in SeqIO.parse(fasta_fname, "fasta"):

        bool=re.search(r'[^ACDEFGHIKLMNPQRSTVWY]',(str(seq_record.seq)).upper())
        if bool:continue
        # seq = re.sub(r'[U]', "C", (str(seq_record.seq)).upper())
        # seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', "", seq)
        seq=(str(seq_record.seq)).upper()
        if len(seq) > min_len:
            if re.search('\|', seq_record.id):
                seq_ids.append(re.search('\|(\w+)\|', seq_record.id).group(1))
            else:
                seq_ids.append(seq_record.id)
            seqs.append(seq)
        else:
            print(seq_record.id, 'Not encoding')
            continue
    return seq_ids, seqs


def encode_select(encode, seqs, start, end, **kwargs):
    """
    :param encode:
    :param seqs:
    :param start:
    :param end:
    :param kwargs: seq_ids, out_fname
    :return:
    """
    feature_matrix, feature_name = [], []
    if encode.split('-')[0] == 'doc2vector':
        # doc2vec parameters
        k, extract_method, vector_size, window, epoch = get_doc2vec_parameters(encode)
        feature_matrix, feature_name, model = doc2vector(seqs, kwargs['seq_ids'], start, end, k,
        extract_method, vector_size, window, epoch)
        model.save('%s_model.pkl' % kwargs['out_fname'])
        return feature_matrix, feature_name
    elif encode.split('-')[0] == 'DPC':
        # DPC parameters
        pass
    else:
        print('%s encoding is not existed!!!' % encode)
        exit()
    return feature_matrix, feature_name


def fasta_to_encoding(fasta_fname, encode, start, end, min_len, out_fname, dtype):
    """
    设置 start 和 end 来设置序列编码区间
    如果 encode是 integer的话，则长度由encode.split('-')[1]来设定
    :param fasta_fname:
    :param encode:
    :param start:
    :param end:
    :param out_fname:
    :param dtype:
    :return:
    """
   
    seq_ids, seqs = extract_seq(fasta_fname, min_len)

    # select the encoding method
    feature_matrix, feature_name = encode_select(encode, seqs, start, end,
    seq_ids=seq_ids, out_fname=out_fname)

    if dtype == 'pkl':
        if re.search('\+', encode):
            feature_dict = feature_matrix
        else:
            feature_dict = dict(zip(seqs, feature_matrix))
        with open('%s.pkl' % out_fname, 'wb') as f:
            print('fname is ', out_fname)
            pickle.dump(feature_dict, f)
            print(feature_name)
            pickle.dump(feature_name, f)
    elif dtype == 'pd':
        feature_matrix = np.asarray(feature_matrix)
        df = pd.DataFrame(feature_matrix, columns=['f_%s' % (i + 1) for i in range(feature_matrix.shape[1])])
        df.insert(0, 'protein_name', seqs)
        f = open('%s.pd' % out_fname, 'w')
        f.write('# %s\n' % '\t'.join(feature_name))
        df.to_csv(f, sep='\t', index=None)
        f.close()


def ac(seqid_seq):
    import os,sys,re,math
    final_lag=30
    r=open("2_normalized_physicochemical_value.txt")
    phychem={}
    for line in r.readlines():
        line=line.strip()
        all7=line.split(" ")
        aa=all7[0]
        for index in range(1,8):
            if index not in phychem:
                phychem[index]={}
                if aa not in phychem[index]:
                    phychem[index][aa]=float(all7[index])
                else:print("error")
            else:
                if aa not in phychem[index]:
                    phychem[index][aa]=float(all7[index])
                else:print("error")
    num=0
    seqids_features={}
    for seqid in seqid_seq:
        seq=seqid_seq[seqid]
        if 'U' in seq:seq=seq.replace('U','')
        ac_encode=[]
        n=len(seq)
        for index in range(1,8):
            sum_aa=0;avg=0
            for eachaa in seq:
                sum_aa+=phychem[index][eachaa]
            avg=sum_aa/n
            for lag in range(1,final_lag+1):
                AC=0
                for i in range(1,(n-lag+1)):
                    xij=phychem[index][seq[i-1]]-avg
                    xilagj=phychem[index][seq[i+lag-1]]-avg
                    AC+=(xij*xilagj)
                num+=1
                AC/=(n-lag)
                ac_encode.append("%.6f"%AC)
        seqids_features[seqid]=ac_encode

    return(seqids_features)



def ct(seqid_seq):
    import os,sys,re
    all={"A":"A","G":"A","V":"A","C":"B","D":"C","E":"C","F":"D","I":"D","L":"D","P":"D","H":"E","N":"E","Q":"E","W":"E","K":"F","R":"F","M":"G","S":"G","T":"G","Y":"G"}
    compond7=["A","B","C","D","E","F","G"]
    set343_h,set343_v={},{}
    for i in compond7:
        for j in compond7:
            for k in compond7:
                if i+j+k not in set343_h:
                    set343_h[i+j+k]=0
                else:
                    print(error)
                if i+j+k not in set343_v:
                    set343_v[i+j+k]=0
                else:
                    print(error)
    seqids_features={}
    for seqid in seqid_seq:
        seq=seqid_seq[seqid]
        if 'U' in seq:seq=seq.replace('U','')
        ct_encode=[]
        i=0;
        for acid_compond1 in set343_h:
            set343_h[acid_compond1]=0
        for acid_compond2 in set343_v:
            set343_v[acid_compond2]=0
        while i+2 <=len(seq)-1:
            acid3=all[seq[i]]+all[seq[i+1]]+all[seq[i+2]]
            set343_h[acid3]+=1
            i+=1
        num=1##参照CKSAAP Ntotal=seqlength-k-1 k为间隔,在此k=1
        for acid in sorted(set343_h.keys()):
            if set343_h[acid]>len(seq):
                print(set343_h[acid],len(seq),line)
            ct_encode.append('%.6f'%(set343_h[acid]/(len(seq)-2)))
            num+=1
        seqids_features[seqid]=ct_encode
    return(seqids_features)

def ld(seqid_seq):
    GROUP = {'A': '1', 'G': '1', 'V': '1',
         'C': '2',
         'D': '3', 'E': '3',
         'I': '4', 'L': '4', 'F': '4', 'P': '4',
         'H': '5', 'N': '5', 'Q': '5', 'W': '5',
         'R': '6', 'K': '6',
         'M': '7', 'T': '7', 'S': '7', 'Y': '7'
    }
    seqids_features={}
    class_keys = [i for i in '1234567']
    class_pair_keys = ['%s%s' % (i, j) for i in class_keys for j in class_keys if i < j]
    pos_keys = ['%s%s' % (i, j) for i in class_keys for j in ['_first', '_25%', '_50%', '_75%', '_last']]
    feature_keys = ['region%s_%s' % (i, j) for i in range(10) for j in class_keys + class_pair_keys + pos_keys]
    # range(10) 代表将序列分为10段

    feature_matrix = []
    for seqid in seqid_seq:
        seq = seqid_seq[seqid]
        seq = ''.join([GROUP[res] for res in seq])
        # C代表在局部序列中每种氨基酸的组成，维度为7
        # D代表相邻两组不同氨基酸所占的比例，维度为6x7/2=21
        # T代表每一组残基的第一个，前25%, 50%, 75%, 100%的残基出现的位置
        div25, div50, div75, div = int(len(seq) / 4), int(len(seq) / 2), int(len(seq) * 3 / 4), len(seq)
        regions = [
            seq[:div25], seq[div25:div50], seq[div50:div75], seq[div75:],
            seq[:div50], seq[div50:], seq[div25:div75], seq[:div75],
            seq[div25:], seq[int(div/8): int(div * 7/8)]  # 中间百分之75
        ]
        sample = []
        for region in regions:
            count_sig_aa = Counter(region)
            C = ['%.6f'%(count_sig_aa[key] * 1.0 / len(region)) for key in class_keys]

            pair_aa = [''.join(sorted(region[i:i+2])) for i in range(len(region) - 1) if region[i] != region[i+1]]
            # pair_aa 为连续两个氨基酸; region[i] ！= region[i+1]说明两个氨基酸不属于同一class; sorted对连续不同两类进行排序
            count_pair_aa = Counter(pair_aa)
            # 如果不存在连续不同类的氨基酸对, 即else 0
            T = ['%.6f'%(count_pair_aa[key] * 1.0 / len(pair_aa)) if len(pair_aa) != 0 else '0' for key in class_pair_keys]

            D = []
            for i in class_keys:
                num1, num25, num50, num75, num100 = 1, int(count_sig_aa[i] / 4), int(count_sig_aa[i] / 2),\
                                                    int(count_sig_aa[i] * 3 / 4), count_sig_aa[i]
                #  num1 .. num100分别是第1位, 25%, ..100%位残基位置
                locs = []
                for num in [num1, num25, num50, num75, num100]:
                    positions = [m.start() + 1 for m in re.finditer(i, region)]
                    if positions:
                        locs.append(['%.6f'%(m.end() * 1.0/len(region)) for m in re.finditer(i, region)][num - 1])
                        # m.start()匹配到m，并记录m的起始位置
                    else:
                        locs.append('0')
                D += locs
            sample += C + T + D
        
        seqids_features[seqid]=sample

    return (seqids_features)


def dpc(seqid_seq):
    aa='ACDEFGHIKLMNPQRSTVWY'
    seqids_features={}
    for seqid in seqid_seq:
        seq=seqid_seq[seqid]
        if 'U' in seq:seq=seq.replace('U','')
        dpc_proportion={}
        dpctmp=[]
        for i in range(20):
            for j in range(20):
                dpc=aa[i]+aa[j]
                dpc_proportion[dpc]=0
        for i in range(len(seq)):
            if i==len(seq)-1:continue
            dpc=seq[i]+seq[i+1]
            dpc_proportion[dpc]+=1
        for dpc in sorted(dpc_proportion):
            dpc_proportion[dpc]/=(len(seq)-1)
            dpctmp.append('%.6f'%dpc_proportion[dpc])

        seqids_features[seqid]=dpctmp
        
    return(seqids_features)

def pro_species(hfile, vfile):
    r=open(hfile)
    hlist, vlist=[], []
    for line in r.readlines():
        line=line.strip()
        hlist.append(line)
    r.close()
    r=open(vfile)
    for line in r.readlines():
        line=line.strip()
        vlist.append(line)
    return(hlist, vlist)

def node2vec_net32(seqid_seq):
    r=open('./Net/hvnet_node2vec32.txt')
    hfile='all_sample_hpro.txt'
    vfile='all_sample_vpro.txt'
    hlist, vlist=pro_species(hfile, vfile)
    seqid_feature={}
    seqids_features={}
    for line in r.readlines():
        line=line.strip()
        pro=line.split()[0]
        feature=line.split()[1:]
        seqid_feature[pro]=feature
    seqids_features={}
    for seqid in seqid_seq:
        if seqid in seqid_feature:seqids_features[seqid]=seqid_feature[seqid]
        else:
            if seqid in hlist:seqids_features[seqid]=seqid_feature['Average_h']
            elif seqid in vlist:seqids_features[seqid]=seqid_feature['Average_v']

    return(seqids_features)

def go64(seqid_seq):
    r=open('./GO/hvgo_node2vec64.txt')
    hfile='all_sample_hpro.txt'
    vfile='all_sample_vpro.txt'
    hlist, vlist=pro_species(hfile, vfile)
    seqid_feature={}
    seqids_features={}
    r.readline()
    for line in r.readlines():
        line=line.strip()
        pro=line.split()[0]
        feature=line.split()[1:]
        seqid_feature[pro]=feature
    seqids_features={}
    for seqid in seqid_seq:
        if seqid in seqid_feature:seqids_features[seqid]=seqid_feature[seqid]
        else:
            if seqid in hlist:seqids_features[seqid]=seqid_feature['Average_h']
            elif seqid in vlist:seqids_features[seqid]=seqid_feature['Average_v']
            else:print(seqid)
    return(seqids_features)

def feature_encode(ecoding,sampleppi_dir,doc2vec_virusshorts,allecoding_proseqfile,encode,start,end, model_dir, complete_infer,modelname,encodename):
    import os

    if ecoding=='doc2vec':
    
        seq_ids,seqs=[],[]
        with open(sampleppi_dir+allecoding_proseqfile) as f1:
            for line in f1.readlines():
                line=line.strip('\n')
                if line[0]=='>': seq_ids.append(line[1:])
                else:seqs.append(line)
              
        f1.close()
        ecoding_dir=sampleppi_dir+'doc2vec/'
        model_fname=model_dir+'interspecies_'+'all'+'-'+encode+'_'+str(start)+'-'+str(end)+'_'+modelname+'_model.pkl'
        encoding_fname=model_dir+'interspecies_'+'all'+'-'+encode+'_'+str(start)+'-'+str(end)+'_'+modelname+'.pkl'
        seqs_features=infer_vector(seq_ids, seqs, start, end, encode, model_fname, encoding_fname, complete_infer)
        for seq in seqs_features:
            tmpfeature=[]
            for each in seqs_features[seq]:
                tmpfeature.append(str(each))
            seqs_features[seq]=tmpfeature
    else:
        seqid_seq={}
        with open(sampleppi_dir+allecoding_proseqfile) as f:
            for line in f.readlines():
                line=line.strip('\n')
                if line[0]=='>':
                    pro=line[1:]
                else:
                    seq=line
                    seqid_seq[pro]=seq
        f.close()
        if ecoding=='ac':
            seqs_features=ac(seqid_seq)           
        elif ecoding=='ct':
            seqs_features=ct(seqid_seq)
        elif ecoding=='dpc':
            seqs_features=dpc(seqid_seq)
        elif ecoding=='net':
            seqs_features=node2vec_net32(seqid_seq)
        elif ecoding=='go':
            seqs_features=go64(seqid_seq)

    return(seqs_features)



def main_doc2vec():
    sampleppi_dir='./'
    k='1' #sys.argv[1]
    window='3' #sys.argv[2]
    vec='256' #sys.argv[3]
    method='2' #sys.argv[5]
    epoch='70' #sys.argv[4]
    for i in range(1,4):
        for m in ['c1','c3']:
            encode='doc2vector-all-'+k+'-'+method+'-'+vec+'-'+window+'-'+epoch
            dir1='../sample/1_10_'+m+'_'+str(i)+'/'
            dir2= '../sample/1_10_'+m+'_'+str(i)+'/'+'doc2vec/1_10_'+m+'_'+str(i)+'/'
            allecoding_proseqfile='sample_'+m+'_'+str(i)+'.fasta'
            print(allecoding_proseqfile)
            doc2vec_model_dir='../embeddings/'
            encodename='doc2vec'  
            modelname='doc2vec_cdhit'
            seqs_features=feature_encode('doc2vec',sampleppi_dir,['all'],allecoding_proseqfile,encode,0,5000,doc2vec_model_dir,False,modelname,encodename)
            file=os.listdir(dir1)
            os.system('mkdir -p '+dir2)
            encode='doc2vector-all-'+k+'-'+method+'-'+vec+'-'+window+'-'+epoch
            for name in file:
                if name not in ['set1','set2','set3','set4','set5','independent_test']:continue
                neg=open(dir1+name)
                w1=open(dir2+name,"w")
                for line in neg.readlines():
                    line=line.strip()
                    whole=line.split("\t")
                    humanid=whole[0]
                    virusid=whole[1]
                    label=whole[2]
                    human_seq=whole[3]
                    virus_seq=whole[4]
                    w1.write(humanid+' '+virusid+" "+label+" "+' '.join(seqs_features[human_seq])+' '+' '.join(seqs_features[virus_seq])+'\n')
                neg.close()
                w1.close()




def main_others(encode):
    import sys
    sampleppi_dir='./'
    #  final encoding: doc2vec_cdhit13256702 k=1 w=3 vector=256 epoch=70, node2vec_net32, go64, combine doc2vec_cdhit13256702node2vec_net32go64
    for i in range(1,4):
        for m in ['c1','c3']:
            dir1='../sample/1_10_'+m+'_'+str(i)+'/'
            dir2= '../sample/1_10_'+m+'_'+str(i)+'/'+encode+'/1_10_'+m+'_'+str(i)+'/'
            allecoding_proseqfile='sample_'+m+'_'+str(i)+'.fasta'
            print(allecoding_proseqfile)
            seqs_features=feature_encode(encode,sampleppi_dir,['all'],allecoding_proseqfile,encode,0,5000,encode,False,encode,encode)
            file=os.listdir(dir1)
            os.system('mkdir -p '+dir2)
            for name in file:
                if name not in ['set1','set2','set3','set4','set5','independent_test']:continue
                id_seq=open(dir1+name)
                w=open(dir2+name,"w")
                for line in id_seq.readlines():     
                    line=line.strip()
                    whole=line.split("\t")
                    humanid=whole[0]
                    virusid=whole[1]
                    label=whole[2]
                    human_seq=whole[3]
                    virus_seq=whole[4]
                    if seqs_features[virusid]=='NA':w.write(humanid+' '+virusid+" "+label+" "+' '.join(seqs_features[humanid])+'\n')
                    else:w.write(humanid+' '+virusid+" "+label+" "+' '.join(seqs_features[humanid])+' '+' '.join(seqs_features[virusid])+'\n')

                                        
                w.close()
                id_seq.close()



def main():
    import sys
    if sys.argv[1]=='doc2vec':
        main_doc2vec()
    else:
        main_others(sys.argv[1])

main()
