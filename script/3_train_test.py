#-*-coding:utf-8 -*-
###导入所需库文件
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.externals import joblib
import joblib
# from sklearn.metrics import roc_curve, auc
import os,sys
from collections import Counter
from sklearn.metrics import roc_curve, auc,roc_auc_score,precision_score, recall_score, f1_score,average_precision_score,accuracy_score

def rf(method, div, algorithm):
    train={"set1234":"set5","set1235":"set4","set1245":"set3","set1345":"set2","set2345":"set1"}
    best_score=0;
    best_params={}
    filedir='1_10_'+div+'/'
    algorithmdir='sample/'+filedir+method+'/'+filedir+algorithm+'/'
    os.system('mkdir -p '+algorithmdir)
    param=open(algorithmdir+"parameters","a")
    sys.stdout=param
    tmpdir='./sample/'+filedir+method+"/"+filedir
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 > '+tmpdir+'set1234')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set5 > '+tmpdir+'set1235')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1245')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1345')
    os.system('cat '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set2345')

    data1=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set1",dtype =str)
    data2=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set2",dtype =str)
    data3=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set3",dtype =str)
    data4=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set4",dtype =str)
    data5=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set5",dtype =str)
    data=np.vstack((data1,data2,data3,data4,data5))
    del data1
    del data2
    del data3
    del data4
    del data5

    y=data[:,2]
    x=data[:,3:]
    del data
    y=y.astype(int)
    x=x.astype(float)

    ##权重影响表现
    counter = Counter(y)##y即y_train训练标签
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}
    parameters = [{'n_estimators':[100,500,1000,1500],  ##默认100。10太小
    'criterion':['entropy','gini'], ##默认gini一般默认就好
    'max_depth':[10,50,100,200], #一般10-100，特征少可以不管，特征多要控制
    # 'oob_score':True, #:即是否采用袋外样本来评估模型的好坏。默认识False。个人推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。
    #师兄让先注释掉'min_samples_split':[2,5,10], ##默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
    #师兄让先注释掉'min_weight_fraction_leaf':[0.0,0.1,0.2,0.3,0.4,0.5]
    # 'max_features':[]
    }]
    clf = GridSearchCV(RandomForestClassifier(class_weight=class_weight), parameters,n_jobs=52,cv=5,scoring='neg_log_loss')
    clf.fit(x,y)
    params=clf.best_params_
    print('best_params_: ',params)
    print('Dataset\tAUC\tPRAUC')
    del clf
    # params={'criterion': 'entropy', 'max_depth': 50, 'n_estimators': 1500}

    clf=RandomForestClassifier(class_weight=class_weight,n_estimators=params["n_estimators"],max_depth=params["max_depth"],criterion=params["criterion"],n_jobs=52)
    modelfinal=clf.fit(x,y)


    del y
    del x

    independent=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/independent_test",dtype =str)
    ppi_validate=independent[:,0:2]
    x_validate=independent[:,3:]
    y_validate=independent[:,2]
    del independent
    y_validate=y_validate.astype(int)
    x_validate=x_validate.astype(float)


    prob_predict_y_validate=clf.predict_proba(x_validate)
    predictions_validate=prob_predict_y_validate[:,1]
    scoreindependent=roc_auc_score(y_validate,predictions_validate)
    pr=average_precision_score(y_validate,predictions_validate)

    print('independent_test_set12345\t'+'%.3f'%scoreindependent+'\t'+'%.3f'%pr)

    ##导出5个独立测试集概率值(+1)
    with open(algorithmdir+"set12345_independent.txt",'a') as f:
        for each in range(len(ppi_validate)):
            f.write('\t'.join(list(ppi_validate[each]))+'\t'+str(y_validate[each])+'\t'+str(predictions_validate[each])+'\n')
    f.close()
    del ppi_validate
    del y_validate
    del x_validate
    #保存模型
    joblib.dump(modelfinal,algorithmdir+'set12345.model')
    del modelfinal
    del clf

    num=1
    average_auc, average_auprc = 0, 0
    for name in sorted(train.keys()):
        train_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+name,dtype =str)
        
        # ppi_train=train_set[:,0:2]
        y_train=train_set[:,2]
        x_train=train_set[:,3:]
        del train_set
        y_train=y_train.astype(int)
        x_train=x_train.astype(float)
        ##权重影响表现
        counter = Counter(y_train)##y即y_train训练标签
        majority = max(counter.values())
        class_weight = {cls: float(majority / count) for cls, count in counter.items()}
        clf=RandomForestClassifier(class_weight=class_weight,n_estimators=params["n_estimators"],max_depth=params["max_depth"],criterion=params["criterion"])#oob_score="True",min_samples_split=best_params["min_samples_split"],min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"]
        model=clf.fit(x_train,y_train)
        
        del y_train
        del x_train


        test_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+train[name],dtype =str)
        ppi_test=test_set[:,0:2]
        y_test=test_set[:,2]
        x_test=test_set[:,3:]
        del test_set
        y_test=y_test.astype(int)
        x_test=x_test.astype(float)
        
        prob_predict_y_test=clf.predict_proba(x_test)
        predictions_test=prob_predict_y_test[:,1]
        # fpr, tpr, _ = roc_curve(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        # roc_auc = auc(fpr, tpr)
        score= roc_auc_score(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        pr=average_precision_score(y_test,predictions_test)
        average_auc+=score
        average_auprc+=pr
        print(name+'\t'+'%.3f'%score+'\t'+'%.3f'%pr)

        ##导出5个测试集概率值(+1)
        with open(algorithmdir+name+".txt",'a') as f1:
            for each1 in range(len(ppi_test)):
                f1.write('\t'.join(list(ppi_test[each1]))+'\t'+str(y_test[each1])+'\t'+str(predictions_test[each1])+'\n')
        f1.close()

        del ppi_test
        del y_test
        del x_test
        del clf

    average_auc/=len(train.keys())
    average_auprc/=len(train.keys())
    print('Average AUC for 5-fold cross-validation: %.3f'%average_auc)
    print('Average APRUC for 5-fold cross-validation: %.3f'%average_auprc)
    param.close()


def lgbm(method, div, algorithm):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import GridSearchCV
    import joblib
    from sklearn.metrics import roc_curve, auc,roc_auc_score,precision_score, recall_score, f1_score,average_precision_score,accuracy_score
    import os,sys
    from collections import Counter
    import lightgbm as lgb
    train={"set1234":"set5","set1235":"set4","set1245":"set3","set1345":"set2","set2345":"set1"}
    best_score=0;
    best_params={}
    filedir='1_10_'+div+'/'
    algorithmdir='sample/'+filedir+method+'/'+filedir+algorithm+'/'
    os.system('mkdir -p '+algorithmdir)
    param=open(algorithmdir+"parameters","a")
    sys.stdout=param
    tmpdir='./sample/'+filedir+method+"/"+filedir
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 > '+tmpdir+'set1234')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set5 > '+tmpdir+'set1235')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1245')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1345')
    os.system('cat '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set2345')

    data1=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set1",dtype =str)
    data2=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set2",dtype =str)
    data3=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set3",dtype =str)
    data4=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set4",dtype =str)
    data5=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set5",dtype =str)
    data=np.vstack((data1,data2,data3,data4,data5))
    del data1
    del data2
    del data3
    del data4
    del data5

    y=data[:,2]
    x=data[:,3:]
    del data
    y=y.astype(int)
    x=x.astype(float)

    ##权重影响表现
    counter = Counter(y)##y即y_train训练标签
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}
    parameters = [{'n_estimators':[100,500,1000,1500],  ##默认100。10太小
    'learning_rate':[0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3],
    #'max_depth':[10,50,100,200],
    'objective':['binary'] #fixed
    }]
    clf = GridSearchCV(lgb.sklearn.LGBMClassifier(class_weight=class_weight), parameters,n_jobs=62,cv=5,scoring='neg_log_loss')
    clf.fit(x,y)
    params=clf.best_params_
    print('best_params_: ',params)
    print('Dataset\tAUC\tPRAUC')
    del clf
    # params={'criterion': 'entropy', 'max_depth': 50, 'n_estimators': 1500}

    clf=lgb.sklearn.LGBMClassifier(class_weight=class_weight,objective=params["objective"],n_estimators=params["n_estimators"],learning_rate=params["learning_rate"],n_jobs=62)
    modelfinal=clf.fit(x,y)


    del y
    del x

    independent=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/independent_test",dtype =str)
    ppi_validate=independent[:,0:2]
    x_validate=independent[:,3:]
    y_validate=independent[:,2]
    del independent
    y_validate=y_validate.astype(int)
    x_validate=x_validate.astype(float)


    prob_predict_y_validate=clf.predict_proba(x_validate)
    predictions_validate=prob_predict_y_validate[:,1]
    scoreindependent=roc_auc_score(y_validate,predictions_validate)
    pr=average_precision_score(y_validate,predictions_validate)

    print('independent_test_set12345\t'+'%.3f'%scoreindependent+'\t'+'%.3f'%pr)

    ##导出5个独立测试集概率值(+1)
    with open(algorithmdir+"set12345_independent.txt",'a') as f:
        for each in range(len(ppi_validate)):
            f.write('\t'.join(list(ppi_validate[each]))+'\t'+str(y_validate[each])+'\t'+str(predictions_validate[each])+'\n')
    f.close()
    del ppi_validate
    del y_validate
    del x_validate
    #保存模型
    joblib.dump(modelfinal,algorithmdir+'set12345.model')
    del modelfinal
    del clf

    num=1
    average_auc, average_auprc = 0, 0
    for name in sorted(train.keys()):
        train_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+name,dtype =str)
        
        # ppi_train=train_set[:,0:2]
        y_train=train_set[:,2]
        x_train=train_set[:,3:]
        del train_set
        y_train=y_train.astype(int)
        x_train=x_train.astype(float)
        ##权重影响表现
        counter = Counter(y_train)##y即y_train训练标签
        majority = max(counter.values())
        class_weight = {cls: float(majority / count) for cls, count in counter.items()}
        clf=lgb.sklearn.LGBMClassifier(class_weight=class_weight,objective=params["objective"],n_estimators=params["n_estimators"],learning_rate=params["learning_rate"],n_jobs=62)
        model=clf.fit(x_train,y_train)
        
        del y_train
        del x_train


        test_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+train[name],dtype =str)
        ppi_test=test_set[:,0:2]
        y_test=test_set[:,2]
        x_test=test_set[:,3:]
        del test_set
        y_test=y_test.astype(int)
        x_test=x_test.astype(float)
        
        prob_predict_y_test=clf.predict_proba(x_test)
        predictions_test=prob_predict_y_test[:,1]
        # fpr, tpr, _ = roc_curve(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        # roc_auc = auc(fpr, tpr)
        score= roc_auc_score(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        pr=average_precision_score(y_test,predictions_test)
        average_auc+=score
        average_auprc+=pr
        print(name+'\t'+'%.3f'%score+'\t'+'%.3f'%pr)

        ##导出5个测试集概率值(+1)
        with open(algorithmdir+name+".txt",'a') as f1:
            for each1 in range(len(ppi_test)):
                f1.write('\t'.join(list(ppi_test[each1]))+'\t'+str(y_test[each1])+'\t'+str(predictions_test[each1])+'\n')
        f1.close()

        del ppi_test
        del y_test
        del x_test
        del clf

    average_auc/=len(train.keys())
    average_auprc/=len(train.keys())
    print('Average AUC for 5-fold cross-validation: %.3f'%average_auc)
    print('Average APRUC for 5-fold cross-validation: %.3f'%average_auprc)
    print('End',time.ctime())
    param.close()


def svm(method, div, algorithm):
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    import joblib
    from sklearn.metrics import roc_curve, auc,roc_auc_score,precision_score, recall_score, f1_score,average_precision_score,accuracy_score
    import os,sys
    from collections import Counter
    from sklearn.svm import SVC
    import random
    train={"set1234":"set5","set1235":"set4","set1245":"set3","set1345":"set2","set2345":"set1"}
    best_score=0;
    best_params={}
    filedir='1_10_'+div+'/'
    algorithmdir='sample/'+filedir+method+'/'+filedir+algorithm+'/'
    os.system('mkdir -p '+algorithmdir)
    param=open(algorithmdir+"parameters","a")
    sys.stdout=param
    tmpdir='./sample/'+filedir+method+"/"+filedir
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 > '+tmpdir+'set1234')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set5 > '+tmpdir+'set1235')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1245')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1345')
    os.system('cat '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set2345')

    data1=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set1",dtype =str)
    data2=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set2",dtype =str)
    data3=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set3",dtype =str)
    data4=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set4",dtype =str)
    data5=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set5",dtype =str)
    data=np.vstack((data1,data2,data3,data4,data5))
    del data1
    del data2
    del data3
    del data4
    del data5

    y=data[:,2]
    x=data[:,3:]
    del data
    y=y.astype(int)
    x=x.astype(float)

    ##权重影响表现
    counter = Counter(y)##y即y_train训练标签
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}
    C_range = [0.1,1]
    #C_range = [0.1,1,10]
    #C_range = np.logspace(-5, 15, 10, base=2)
    gamma_range = [0.01,0.1,1]
    #gamma_range = [0.01,0.1,1,10]
    #gamma_range = np.logspace(3, -15, 9, base=2)
    parameters = [{  
    'C':list(C_range),
    'gamma':list(gamma_range),

    }]
    clf = GridSearchCV(SVC(class_weight=class_weight,probability=True), parameters,n_jobs=45,cv=5,scoring='neg_log_loss')
    clf.fit(x,y)
    params=clf.best_params_
    print('best_params_: ',params)
    print('Dataset\tAUC\tPRAUC')
    del clf
    # params={'criterion': 'entropy', 'max_depth': 50, 'n_estimators': 1500}

    clf=SVC(class_weight=class_weight,probability=True,C=params["C"],gamma=params["gamma"])
    modelfinal=clf.fit(x,y)


    del y
    del x

    independent=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/independent_test",dtype =str)
    ppi_validate=independent[:,0:2]
    x_validate=independent[:,3:]
    y_validate=independent[:,2]
    del independent
    y_validate=y_validate.astype(int)
    x_validate=x_validate.astype(float)


    prob_predict_y_validate=clf.predict_proba(x_validate)
    predictions_validate=prob_predict_y_validate[:,1]
    scoreindependent=roc_auc_score(y_validate,predictions_validate)
    pr=average_precision_score(y_validate,predictions_validate)

    print('independent_test_set12345\t'+'%.3f'%scoreindependent+'\t'+'%.3f'%pr)

    ##导出5个独立测试集概率值(+1)
    with open(algorithmdir+"set12345_independent.txt",'a') as f:
        for each in range(len(ppi_validate)):
            f.write('\t'.join(list(ppi_validate[each]))+'\t'+str(y_validate[each])+'\t'+str(predictions_validate[each])+'\n')
    f.close()
    del ppi_validate
    del y_validate
    del x_validate
    #保存模型
    joblib.dump(modelfinal,algorithmdir+'set12345.model')
    del modelfinal
    del clf

    num=1
    average_auc, average_auprc = 0, 0
    for name in sorted(train.keys()):
        train_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+name,dtype =str)
        
        # ppi_train=train_set[:,0:2]
        y_train=train_set[:,2]
        x_train=train_set[:,3:]
        del train_set
        y_train=y_train.astype(int)
        x_train=x_train.astype(float)
        ##权重影响表现
        counter = Counter(y_train)##y即y_train训练标签
        majority = max(counter.values())
        class_weight = {cls: float(majority / count) for cls, count in counter.items()}
        clf=SVC(class_weight=class_weight,probability=True,C=params["C"],gamma=params["gamma"])
        model=clf.fit(x_train,y_train)
        
        del y_train
        del x_train


        test_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+train[name],dtype =str)
        ppi_test=test_set[:,0:2]
        y_test=test_set[:,2]
        x_test=test_set[:,3:]
        del test_set
        y_test=y_test.astype(int)
        x_test=x_test.astype(float)
        
        prob_predict_y_test=clf.predict_proba(x_test)
        predictions_test=prob_predict_y_test[:,1]
        # fpr, tpr, _ = roc_curve(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        # roc_auc = auc(fpr, tpr)
        score= roc_auc_score(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        pr=average_precision_score(y_test,predictions_test)
        average_auc+=score
        average_auprc+=pr
        print(name+'\t'+'%.3f'%score+'\t'+'%.3f'%pr)

        ##导出5个测试集概率值(+1)
        with open(algorithmdir+name+".txt",'a') as f1:
            for each1 in range(len(ppi_test)):
                f1.write('\t'.join(list(ppi_test[each1]))+'\t'+str(y_test[each1])+'\t'+str(predictions_test[each1])+'\n')
        f1.close()

        del ppi_test
        del y_test
        del x_test
        del clf

    average_auc/=len(train.keys())
    average_auprc/=len(train.keys())
    print('Average AUC for 5-fold cross-validation: %.3f'%average_auc)
    print('Average APRUC for 5-fold cross-validation: %.3f'%average_auprc)
    param.close()

def mlp(method, div, algorithm):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout#防止过拟合
    from keras import optimizers
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve, auc,roc_auc_score,precision_score, recall_score, f1_score,average_precision_score,accuracy_score
    from collections import Counter
    import sys
    import joblib
    train={"set1234":"set5","set1235":"set4","set1245":"set3","set1345":"set2","set2345":"set1"}
    best_score=0;
    best_params={}
    filedir='1_10_'+div+'/'
    algorithmdir='sample/'+filedir+method+'/'+filedir+algorithm+'/'
    os.system('mkdir -p '+algorithmdir)
    param=open(algorithmdir+"parameters","a")
    sys.stdout=param
    tmpdir='./sample/'+filedir+method+"/"+filedir
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 > '+tmpdir+'set1234')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set5 > '+tmpdir+'set1235')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set2 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1245')
    os.system('cat '+tmpdir+'set1 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set1345')
    os.system('cat '+tmpdir+'set2 '+tmpdir+'set3 '+tmpdir+'set4 '+tmpdir+'set5 > '+tmpdir+'set2345')

    data1=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set1",dtype =str)
    data2=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set2",dtype =str)
    data3=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set3",dtype =str)
    data4=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set4",dtype =str)
    data5=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/set5",dtype =str)
    data=np.vstack((data1,data2,data3,data4,data5))
    del data1
    del data2
    del data3
    del data4
    del data5

    y=data[:,2]
    x=data[:,3:]
    
    y=y.astype(int)
    x=x.astype(float)

    ##权重影响表现
    counter = Counter(y)##y即y_train训练标签
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}

    #定义模型
    model=Sequential()
    #输入层有len(alltrainset[1,:])-3即128个神经元；隐藏层有128/256个神经元，激活函数为relu
    #此处为第一个隐藏层，
    model.add(Dense(128,input_dim=len(data[1,:])-3,activation='relu'))
    model.add(Dropout(0.3))
    #此处为第二个隐藏层
    model.add(Dense(64,input_dim=len(data[1,:])-3,activation='relu'))
    model.add(Dropout(0.3))

    del data

    #输出层有1个神经元，激活函数为sigmoid
    model.add(Dense(1,activation='sigmoid'))

    #编译模型
    #模型采用的优化器和损失函数类型分别为：adam和binary_crossentropy，训练周期为100，每次数据量为10。
    adam=optimizers.adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    #训练模型 nb_epoch训练周期；batch_size每次数据量
    history=model.fit(x,y,epochs=120,batch_size=64,class_weight=class_weight)
    #learning_rate: 0.001 0.0001


    del y
    del x

    independent=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+"/independent_test",dtype =str)
    ppi_validate=independent[:,0:2]
    x_validate=independent[:,3:]
    y_validate=independent[:,2]
    del independent
    y_validate=y_validate.astype(int)
    x_validate=x_validate.astype(float)


    prob_predict_y_validate=model.predict_proba(x_validate)
    predictions_validate=prob_predict_y_validate[:,1]
    scoreindependent=roc_auc_score(y_validate,predictions_validate)
    pr=average_precision_score(y_validate,predictions_validate)

    print('independent_test_set12345\t'+'%.3f'%scoreindependent+'\t'+'%.3f'%pr)

    ##导出5个独立测试集概率值(+1)
    with open(algorithmdir+"set12345_independent.txt",'a') as f:
        for each in range(len(ppi_validate)):
            f.write('\t'.join(list(ppi_validate[each]))+'\t'+str(y_validate[each])+'\t'+str(predictions_validate[each])+'\n')
    f.close()
    del ppi_validate
    del y_validate
    del x_validate
    #保存模型
    joblib.dump(history,algorithmdir+'set12345.model')

    num=1
    average_auc, average_auprc = 0, 0
    for name in sorted(train.keys()):
        train_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+name,dtype =str)
        
        # ppi_train=train_set[:,0:2]
        y_train=train_set[:,2]
        x_train=train_set[:,3:]
        del train_set
        y_train=y_train.astype(int)
        x_train=x_train.astype(float)
        ##权重影响表现
        counter = Counter(y_train)##y即y_train训练标签
        majority = max(counter.values())
        class_weight = {cls: float(majority / count) for cls, count in counter.items()}
        #定义模型
        model=Sequential()
        #输入层有len(train_set[1,:])-3即128个神经元；隐藏层有128/256个神经元，激活函数为relu
        model.add(Dense(128,input_dim=len(train_set[1,:])-3,activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(64,input_dim=len(train_set[1,:])-3,activation='relu'))
        model.add(Dropout(0.3))

        del train_set
        #输出层有1个神经元，激活函数为sigmoid
        model.add(Dense(1,activation='sigmoid'))

        #编译模型
        #模型采用的优化器和损失函数类型分别为：adam和binary_crossentropy，训练周期为100，每次数据量为10。
        adam=optimizers.adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        #训练模型 nb_epoch训练周期；batch_size每次数据量
        history=model.fit(x_train,y_train,epochs=120,batch_size=64,class_weight=class_weight)
        #learning_rate: 0.001 0.0001

        del y_train
        del x_train


        test_set=np.genfromtxt("./sample/"+filedir+method+"/"+filedir+train[name],dtype =str)
        ppi_test=test_set[:,0:2]
        y_test=test_set[:,2]
        x_test=test_set[:,3:]
        del test_set
        y_test=y_test.astype(int)
        x_test=x_test.astype(float)
        
        prob_predict_y_test=model.predict_proba(x_test)
        predictions_test=prob_predict_y_test[:,1]
        # fpr, tpr, _ = roc_curve(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        # roc_auc = auc(fpr, tpr)
        score= roc_auc_score(y_test, predictions_test)##如果使用Python的话， 可以直接调用sklearn.metrics.roc_curve来计算坐标点和阈值。fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)。
        pr=average_precision_score(y_test,predictions_test)
        average_auc+=score
        average_auprc+=pr
        print(name+'\t'+'%.3f'%score+'\t'+'%.3f'%pr)

        ##导出5个测试集概率值(+1)
        with open(algorithmdir+name+".txt",'a') as f1:
            for each1 in range(len(ppi_test)):
                f1.write('\t'.join(list(ppi_test[each1]))+'\t'+str(y_test[each1])+'\t'+str(predictions_test[each1])+'\n')
        f1.close()

        del ppi_test
        del y_test
        del x_test
        del clf

    average_auc/=len(train.keys())
    average_auprc/=len(train.keys())
    print('Average AUC for 5-fold cross-validation: %.3f'%average_auc)
    print('Average APRUC for 5-fold cross-validation: %.3f'%average_auprc)
    param.close()

def main():
    import time

    method=sys.argv[1]# doc2vec_cdhit13256702
    div=sys.argv[2] # c3_1
    algorithm=sys.argv[3] # lgbm

    print('Start',time.ctime())
    if algorithm=='rf':
        rf(method, div, algorithm)
    elif algorithm=='lgbm':
        lgbm(method, div, algorithm)
    elif algorithm=='svm':
        svm(method, div, algorithm)
    elif algorithm=='mlp':
        mlp(method, div, algorithm)

if __name__=='__main__':
    main()



