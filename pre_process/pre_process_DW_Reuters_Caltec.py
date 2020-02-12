'''
Created on Jan 30, 2019

@author: eb
'''

import numpy
import random
from numpy.linalg import norm

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    np.random.seed(1364)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#############################################################
############### Data Loaders for DW Dataset #################

#@mem.cache
def load_dw_dataset_en(train=True):
    if train:
        data = load_svmlight_file('./data/dw/trainVecMatrix_en_s', n_features=20000, multilabel=False)
    else:
        data = load_svmlight_file('./data/dw/valVecMatrix_en_s', n_features=20000, multilabel=False)
    X = data[0].toarray()
    y = data[1]
    return X, y

#@mem.cache
def load_dw_dataset_ru(train=True):
    if train:
        data = load_svmlight_file('./data/dw/trainVecMatrix_ru_s', n_features=20000, multilabel=False)
    else:
        data = load_svmlight_file('./data/dw/valVecMatrix_ru_s', n_features=20000, multilabel=False)
    X = data[0].toarray()
    y = data[1]
    return X, y

#@mem.cache
def load_dw_dataset_fr(train=True):
    if train:
        data = load_svmlight_file('./data/dw/trainVecMatrix_fr_s', n_features=20000, multilabel=False)
    else:
        data = load_svmlight_file('./data/dw/valVecMatrix_fr_s', n_features=20000, multilabel=False)
    X = data[0].toarray()
    y = data[1]
    return X, y
    
    
########################################################
############### Data Loaders for Reuters Dataset #######

# Consistent with the literature spain is the target and all other four languages are source
# Convert labels to numbers: CCAT= 1, ECAT= 2, GCAT = 3, C15 = 4, E21 = 5, M11 = 6

def load_reuters_dataset(lang='EN'):
    data = load_svmlight_file('/home/eb/DA/dataset/reuters/Index_' + lang + '-'+lang , multilabel=False, zero_based=False)
    
    X = data[0].toarray()
    y = data[1]
    return X, y

# def load_reuters_dataset_random(lang='EN', use_PCA=False):
#     if use_PCA:
#         data = load_svmlight_file('./data/reuters_PCA/sampled_' + lang.upper(), multilabel=False)
#     else:
#         data = load_svmlight_file('./data/reuters/sampled_' + lang.upper(), multilabel=False)
#     
#     X = data[0].toarray()
#     y = data[1]
#     y_0based = y -1
#     return X, y_0based

def load_reuters_dataset_random(lang='EN', dim_red='PCA', dim=0):
    data = load_svmlight_file('./data/reuters/sampled_' + lang.upper(), multilabel=False)
    X = data[0].toarray()
    y = data[1]
    y_0based = y -1
    if dim == 0:
        return X, y_0based
    elif dim_red == 'PCA':
        pca = PCA(n_components= dim)
        pca.fit(X)
        X_transformed = pca.transform(X)
        return X_transformed, y_0based
    elif dim_red =='LDA':
        lda = LinearDiscriminantAnalysis(n_components= dim)
        lda.fit(X, y)
        X_transformed = lda.transform(X)
        return X_transformed, y_0based

def load_reuters_dataset_random_10labels(lang='EN', dim_red='PCA', dim=0):
    data = load_svmlight_file('./data/reuters_10labels/sampled_' + lang.upper(), multilabel=False)
    X = data[0].toarray()
    y = data[1]
    y_0based = y -1
    if dim == 0:
        return X, y_0based
    elif dim_red == 'PCA':
        pca = PCA(n_components= dim)
        pca.fit(X)
        X_transformed = pca.transform(X)
        return X_transformed, y_0based
    elif dim_red =='LDA':
        lda = LinearDiscriminantAnalysis(n_components= dim)
        lda.fit(X, y)
        X_transformed = lda.transform(X)
        return X_transformed, y_0based     
        
    
''' Only needed to be called once to sample the data.'''
def sample_reuters_dataset_random(lang='EN'):
# Jingjing Li et al., 2018 randomly selects 100 from each category for source and 500 for target

    if lang.lower() =="sp":
        # select 500 randomly
        data, y = load_reuters_dataset("SP")
        # append the data and labels
        l = y.reshape(-1,1)
        l_data = numpy.concatenate((l, data), axis=1)
        numpy.random.seed(1364)
        l_data_500 = l_data[numpy.random.choice(l_data.shape[0], 500, replace=False), :]
        sampled = l_data_500

    elif lang.lower() == "en" or lang.lower() == "fr" or lang.lower()== "gr" or lang.lower() == "it":
    
        ############### select 100 from each category for the identified language
        data, y = load_reuters_dataset(lang)
        # append the data and labels
        l = y.reshape(-1,1)
        l_data = numpy.concatenate((l, data), axis=1)
        
        l_data_CCAT = l_data[numpy.where(l_data[:,0] == 1.0)]
        l_data_ECAT = l_data[numpy.where(l_data[:,0] == 2.0)]
        l_data_GCAT = l_data[numpy.where(l_data[:,0] == 3.0)]
        l_data_C15 = l_data[numpy.where(l_data[:,0] == 4.0)]
        l_data_E21 = l_data[numpy.where(l_data[:,0] == 5.0)]
        l_data_M11 = l_data[numpy.where(l_data[:,0] == 6.0)]
         
        numpy.random.seed(1364)
        l_data_CCAT_100 = l_data_CCAT[numpy.random.choice(l_data_CCAT.shape[0], 100, replace=False), :]  
        l_data_ECAT_100 = l_data_ECAT[numpy.random.choice(l_data_ECAT.shape[0], 100, replace=False), :]
        l_data_GCAT_100 = l_data_GCAT[numpy.random.choice(l_data_GCAT.shape[0], 100, replace=False), :]
        l_data_C15_100 = l_data_C15[numpy.random.choice(l_data_C15.shape[0], 100, replace=False), :]
        l_data_E21_100 = l_data_E21[numpy.random.choice(l_data_E21.shape[0], 100, replace=False), :]
        l_data_M11_100 = l_data_M11[numpy.random.choice(l_data_M11.shape[0], 100, replace=False), :]
    
        sampled = numpy.concatenate((l_data_CCAT_100, l_data_ECAT_100, l_data_GCAT_100, l_data_C15_100, l_data_E21_100, l_data_M11_100), axis=0)
        
    else:
        print "Either the language does not exist or sth went wrong!"
        return
    
    #separate labels
    sampled_data = sampled[:,1:] 
    sampled_y = sampled[:,0]
    
    # write sample in svm_light format
    dump_svmlight_file(sampled_data, sampled_y, "./data/reuters/sampled_"+lang.upper(), multilabel=False)
    print lang + ' was sampled and saved!'

    
''' Only needed to be called once to sample the data.IT IS WRONG TO RUN THE PCA BEFORE SAMPLING, so we reduce to high dimensions (10k) ''' 
def sample_reuters_dataset_random_PCABeforeSampling(lang='EN', latent_dim=1024):
# Jingjing Li et al., 2018 randomly selects 100 from each category for source and 500 for target

    if lang.lower() =="sp":
        # select 500 randomly
        data, y = load_reuters_dataset("SP")
        
        pca = PCA(n_components=latent_dim)
        pca.fit(data)
        data = pca.transform(data)
    
        # append the data and labels
        l = y.reshape(-1,1)
        l_data = numpy.concatenate((l, data), axis=1)
        numpy.random.seed(1364)
        l_data_500 = l_data[numpy.random.choice(l_data.shape[0], 500, replace=False), :]
        sampled = l_data_500

    elif lang.lower() == "en" or lang.lower() == "fr" or lang.lower()== "gr" or lang.lower() == "it":
    
        ############### select 100 from each category for the identified language
        data, y = load_reuters_dataset(lang)
        
        pca = PCA(n_components=latent_dim)
        pca.fit(data)
        data = pca.transform(data)
        
        # append the data and labels
        l = y.reshape(-1,1)
        l_data = numpy.concatenate((l, data), axis=1)
        
        l_data_CCAT = l_data[numpy.where(l_data[:,0] == 1.0)]
        l_data_ECAT = l_data[numpy.where(l_data[:,0] == 2.0)]
        l_data_GCAT = l_data[numpy.where(l_data[:,0] == 3.0)]
        l_data_C15 = l_data[numpy.where(l_data[:,0] == 4.0)]
        l_data_E21 = l_data[numpy.where(l_data[:,0] == 5.0)]
        l_data_M11 = l_data[numpy.where(l_data[:,0] == 6.0)]
         
        numpy.random.seed(1364)
        l_data_CCAT_100 = l_data_CCAT[numpy.random.choice(l_data_CCAT.shape[0], 100, replace=False), :]  
        l_data_ECAT_100 = l_data_ECAT[numpy.random.choice(l_data_ECAT.shape[0], 100, replace=False), :]
        l_data_GCAT_100 = l_data_GCAT[numpy.random.choice(l_data_GCAT.shape[0], 100, replace=False), :]
        l_data_C15_100 = l_data_C15[numpy.random.choice(l_data_C15.shape[0], 100, replace=False), :]
        l_data_E21_100 = l_data_E21[numpy.random.choice(l_data_E21.shape[0], 100, replace=False), :]
        l_data_M11_100 = l_data_M11[numpy.random.choice(l_data_M11.shape[0], 100, replace=False), :]
    
        sampled = numpy.concatenate((l_data_CCAT_100, l_data_ECAT_100, l_data_GCAT_100, l_data_C15_100, l_data_E21_100, l_data_M11_100), axis=0)
        
    else:
        print "Either the language does not exist or sth went wrong!"
        return
    
    #separate labels
    sampled_data = sampled[:,1:] 
    sampled_y = sampled[:,0]
    
    # write sample in svm_light format
    dump_svmlight_file(sampled_data, sampled_y, "./data/reuters_PCA/PCA_"+lang.upper()+"_"+ str(latent_dim), multilabel=False)
    print lang + ' was sampled and saved!'


def sample_reuters_dataset_random_LDABeforeSampling(lang='EN', latent_dim=1024):
# Jingjing Li et al., 2018 randomly selects 100 from each category for source and 500 for target

    if lang.lower() =="sp":
        # select 500 randomly
        data, y = load_reuters_dataset("SP")
        
        lda = LinearDiscriminantAnalysis(n_components=latent_dim)
        lda.fit(data, y)
        data = lda.transform(data)
    
        # append the data and labels
        l = y.reshape(-1,1)
        l_data = numpy.concatenate((l, data), axis=1)
        numpy.random.seed(1364)
        l_data_500 = l_data[numpy.random.choice(l_data.shape[0], 500, replace=False), :]
        sampled = l_data_500

    elif lang.lower() == "en" or lang.lower() == "fr" or lang.lower()== "gr" or lang.lower() == "it":
    
        ############### select 100 from each category for the identified language
        data, y = load_reuters_dataset(lang)
        
        pca = PCA(n_components=latent_dim)
        pca.fit(data)
        data = pca.transform(data)
        
        # append the data and labels
        l = y.reshape(-1,1)
        l_data = numpy.concatenate((l, data), axis=1)
        
        l_data_CCAT = l_data[numpy.where(l_data[:,0] == 1.0)]
        l_data_ECAT = l_data[numpy.where(l_data[:,0] == 2.0)]
        l_data_GCAT = l_data[numpy.where(l_data[:,0] == 3.0)]
        l_data_C15 = l_data[numpy.where(l_data[:,0] == 4.0)]
        l_data_E21 = l_data[numpy.where(l_data[:,0] == 5.0)]
        l_data_M11 = l_data[numpy.where(l_data[:,0] == 6.0)]
         
        numpy.random.seed(1364)
        l_data_CCAT_100 = l_data_CCAT[numpy.random.choice(l_data_CCAT.shape[0], 100, replace=False), :]  
        l_data_ECAT_100 = l_data_ECAT[numpy.random.choice(l_data_ECAT.shape[0], 100, replace=False), :]
        l_data_GCAT_100 = l_data_GCAT[numpy.random.choice(l_data_GCAT.shape[0], 100, replace=False), :]
        l_data_C15_100 = l_data_C15[numpy.random.choice(l_data_C15.shape[0], 100, replace=False), :]
        l_data_E21_100 = l_data_E21[numpy.random.choice(l_data_E21.shape[0], 100, replace=False), :]
        l_data_M11_100 = l_data_M11[numpy.random.choice(l_data_M11.shape[0], 100, replace=False), :]
    
        sampled = numpy.concatenate((l_data_CCAT_100, l_data_ECAT_100, l_data_GCAT_100, l_data_C15_100, l_data_E21_100, l_data_M11_100), axis=0)
        
    else:
        print "Either the language does not exist or sth went wrong!"
        return
    
    #separate labels
    sampled_data = sampled[:,1:] 
    sampled_y = sampled[:,0]
    
    # write sample in svm_light format
    dump_svmlight_file(sampled_data, sampled_y, "./data/reuters_LDA/LDA_"+lang.upper()+"_"+ str(latent_dim), multilabel=False)
    print lang + ' was sampled and saved!'



##########################################################
###Data Loaders for Office-caltech Dataset (SURF-Decaf)###

# Representation: SURF, DECAF; domain: amazon, webcam, dslr, caltec10


def load_officecaltec_dataset(representation='SURF', domain='amazon'):
    
    if (representation.lower() == 'surf'):
        mat_contents = sio.loadmat("./data/office-caltec/surf_zscore/" + domain.lower()+ "_zscore_SURF_L10.mat")
        if (domain == 'dslr'):
            X = mat_contents['Xs']
            y = mat_contents['Ys']
        else:
            X = mat_contents['Xt']
            y = mat_contents['Yt']
        # normalize each row by l2 (surf is z-scored but not normalized to unit)
        row_norms = norm(X, axis=1, ord=2) # length of each row
        X_normed = X.astype(numpy.float) / row_norms[:,None] # devide by length of each row
        y_0based = y -1 # make it 0-indexed
        return unison_shuffled_copies(X_normed, y_0based) # shuffle simultaneously
    elif (representation.lower() == 'decaf'):
        mat_contents = sio.loadmat("./data/office-caltec/decaf6/" + domain.lower()+ "_decaf.mat")
        X = mat_contents['feas']
        y = mat_contents['labels']
        # normalize each row by l2 (Decaf is not normalized to unit)
        row_norms = norm(X, axis=1, ord=2)
        X_normed = X.astype(numpy.float) / row_norms[:,None]
        y_0based = y-1
        return unison_shuffled_copies(X_normed, y_0based) # shuffle simultaneously
    else:
        print "Wrong representation was given!"
        return
    

def load_officecaltec_dataset_PCA(representation='SURF', domain='amazon', PCA_dim =0):
    X, y = load_officecaltec_dataset(representation,domain)
    if PCA_dim ==0:
        return X, y
    else:
        pca = PCA(n_components=PCA_dim)
        pca.fit(X)
        return pca.transform(X), y



######################################################
#### Data preprocessing using Keras ##################
 

# # For efficiency of loading sparse data
# from joblib import Memory
# mem = Memory("./mycache")
#   
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
# from keras.preprocessing import text
# 
#  
# ######## English
# trainPosFile = open("/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train-pos.txt", "r")
# trainNegFile = open("/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train-neg.txt", "r")
#    
# trainTexts=[]
# for l in trainPosFile:
#     trainTexts.append(l)
# trainPosFile.close()
# for l in trainNegFile:
#     trainTexts.append(l)
# trainNegFile.close()
#    
# print trainTexts[0]
# print ('train set size is: ' +str(len(trainTexts)))
#    
# ##Build the labels (Pos=1, Neg=0)
# y_train=[]
# with open('/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train.cat','r') as f:
#     for line in f:
#         if line.strip() == "pos":
#             y_train.append(1)
#         else:
#             y_train.append(0)
# print ('The size of training labels is: ' + str(len(y_train)))
# y_train = numpy.array(y_train)
#    
# #### Build unlabeled data - 58893
# unlabeledTexts = []
# with open('/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/testAll.txt','r') as f:
#     for line in f:
#         unlabeledTexts.append(line)
#         
# print ('unlabele set size is: ' + str(len(unlabeledTexts)))
#    
# #### Build validation (test) data  - 132
# valPosFile = open("/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test-pos.txt", "r")
# valNegFile = open("/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test-neg.txt", "r")
#    
# valTexts=[]
# for l in valPosFile:
#     valTexts.append(l)
# valPosFile.close()
# for l in valNegFile:
#     valTexts.append(l)
# valNegFile.close()
#    
# print ('validation set size is: ' +str(len(valTexts)))
#    
# y_val=[]
# with open('/home/eb/dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test.cat','r') as f:
#     for line in f:
#         if line.strip() == "pos":
#             y_val.append(1)
#         else:
#             y_val.append(0)
# print ('The size of validation labels is: ' + str(len(y_val)))
# y_val = numpy.array(y_val)
#    
#    
# # Build an indexed sequence for each repument
# vocabSize_Eng = 20000
# tokenizer_Eng = text.Tokenizer(num_words=vocabSize_Eng,
#                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#                    lower=True,
#                    split=" ",
#                    char_level=False)
# # Build the word to int mapping based on the training data
# tokenizer_Eng.fit_on_texts(trainTexts)
#   
# # Build the sequence (Keras built-in) for LSTM
# trainTextsSeq = tokenizer_Eng.texts_to_sequences(trainTexts)
# print (trainTextsSeq[0])
# print (len(trainTextsSeq))
# trainTextsSeq = numpy.array(trainTextsSeq)
#   
#    
# valTextsSeq = tokenizer_Eng.texts_to_sequences(valTexts)
# print (valTextsSeq[0])
# print (len(valTextsSeq))
# valTextsSeq = numpy.array(valTextsSeq)
#    
# # for non-sequence vectorization such as tfidf --> SVM
# trainVecMatrix = tokenizer_Eng.sequences_to_matrix(trainTextsSeq, mode='tfidf')
# #print (trainVecMatrix)
# print ('training vector length: '+str(len(trainVecMatrix)))
# print ('training vector columns: '+str(len(trainVecMatrix[0])))
#    
# valVecMatrix = tokenizer_Eng.sequences_to_matrix(valTextsSeq, mode='tfidf')
# print ('validation vector length: '+str(len(valVecMatrix)))
# print ('validation vector columns: '+str(len(valVecMatrix[0])))
#   
# dump_svmlight_file(trainVecMatrix, y_train, "trainVecMatrix_en_s", zero_based=True, multilabel=False)
#   
# dump_svmlight_file(valVecMatrix, y_val, "valVecMatrix_en_s", zero_based=True, multilabel=False)
#     
#  
# ######## Russian
# ################
# trainPosFile_Rus = open("/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-Rus-pos.txt", "r")
# trainNegFile_Rus = open("/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-Rus-neg.txt", "r")
#     
# trainTexts_Rus=[]
# for l in trainPosFile_Rus:
#     trainTexts_Rus.append(l)
# trainPosFile_Rus.close()
# for l in trainNegFile_Rus:
#     trainTexts_Rus.append(l)
# trainNegFile_Rus.close()
#     
# print trainTexts_Rus[0]
# print ('train set size is: ' +str(len(trainTexts_Rus)))
#     
# ##Build the labels (Pos=1, Neg=0)
# y_train_Rus=[]
# with open('/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-RUS-train.cat','r') as f:
#     for line in f:
#         if line.strip() == "pos":
#             y_train_Rus.append(1)
#         else:
#             y_train_Rus.append(0)
# print ('The size of training labels is: ' + str(len(y_train_Rus)))
# y_train_Rus = numpy.array(y_train_Rus) 
#     
# #### Build validation (test) data  - 55 + 117 = 172
# valPosFile_Rus = open("/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-Rus-pos.txt", "r")
# valNegFile_Rus = open("/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-Rus-neg.txt", "r")
#     
# valTexts_Rus=[]
# for l in valPosFile_Rus:
#     valTexts_Rus.append(l)
# valPosFile_Rus.close()
# for l in valNegFile_Rus:
#     valTexts_Rus.append(l)
# valNegFile_Rus.close()
#     
# print ('validation set size is: ' +str(len(valTexts_Rus)))
#     
# y_val_Rus=[]
# with open('/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-RUS-test.cat','r') as f:
#     for line in f:
#         if line.strip() == "pos":
#             y_val_Rus.append(1)
#         else:
#             y_val_Rus.append(0)
# print ('The size of validation labels is: ' + str(len(y_val_Rus)))
# y_val_Rus = numpy.array(y_val_Rus)
#     
# # Build an indexed sequence for each repument
# vocabSize_Rus = 20000
#     
# tokenizer_Rus = text.Tokenizer(num_words=vocabSize_Rus,
#                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#                    lower=True,
#                    split=" ",
#                    char_level=False)
#     
# # Build the word to int mapping based on the training data
# tokenizer_Rus.fit_on_texts(trainTexts_Rus)
#     
# # Build the sequence (Keras built-in) for LSTM
# trainTextsSeq_Rus = tokenizer_Rus.texts_to_sequences(trainTexts_Rus)
# print (trainTextsSeq_Rus[0])
# print (len(trainTextsSeq_Rus))
# trainTextsSeq_Rus = numpy.array(trainTextsSeq_Rus)
#    
#     
# valTextsSeq_Rus = tokenizer_Rus.texts_to_sequences(valTexts_Rus)
# print (valTextsSeq_Rus[0])
# print (len(valTextsSeq_Rus))
# valTextsSeq_Rus = numpy.array(valTextsSeq_Rus)
#     
# # for non-sequence vectorization such as tfidf --> SVM
# trainVecMatrix_Rus = tokenizer_Rus.sequences_to_matrix(trainTextsSeq_Rus, mode='tfidf')
# #print (trainVecMatrix)
# print ('training vector length: '+str(len(trainVecMatrix_Rus)))
# print ('training vector columns: '+str(len(trainVecMatrix_Rus[0])))
#     
# valVecMatrix_Rus = tokenizer_Rus.sequences_to_matrix(valTextsSeq_Rus, mode='tfidf')
# print ('validation vector length: '+str(len(valVecMatrix_Rus)))
# print ('validation vector columns: '+str(len(valVecMatrix_Rus[0])))
#    
# dump_svmlight_file(trainVecMatrix_Rus, y_train_Rus, "trainVecMatrix_ru_s", zero_based=True, multilabel=False)
#    
# dump_svmlight_file(valVecMatrix_Rus, y_val_Rus, "valVecMatrix_ru_s", zero_based=True, multilabel=False)
#  
#  
# ######## French
# ################
# #### Build train data (list of texts) - 380
# trainPosFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/train/DNM-Fr-train-Pos.txt", "r")
# trainNegFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/train/DNM-Fr-train-Neg.txt", "r")
#   
# trainTexts_Fr=[]
# for l in trainPosFile_Fr:
#     trainTexts_Fr.append(l)
# trainPosFile_Fr.close()
# for l in trainNegFile_Fr:
#     trainTexts_Fr.append(l)
# trainNegFile_Fr.close()
#   
# print trainTexts_Fr[0]
# print ('train set size is: ' +str(len(trainTexts_Fr)))
#   
# ##Build the labels (Pos=1, Neg=0)
# y_train_Fr=[]
# with open('/home/eb/dnm_data/MultiLingual/French-Labeled/train/DNM-Fr-train.cat','r') as f:
#     for line in f:
#         if line.strip() == "pos":
#             y_train_Fr.append(1)
#         else:
#             y_train_Fr.append(0)
# print ('The size of training labels is: ' + str(len(y_train_Fr)))
# y_train_Fr = numpy.array(y_train_Fr) 
#   
# #### Build validation (test) data  - 55 + 117 = 172
# valPosFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-Fr-test-Pos.txt", "r")
# valNegFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-Fr-test-Neg.txt", "r")
#   
# valTexts_Fr=[]
# for l in valPosFile_Fr:
#     valTexts_Fr.append(l)
# valPosFile_Fr.close()
# for l in valNegFile_Fr:
#     valTexts_Fr.append(l)
# valNegFile_Fr.close()
#   
# print ('validation set size is: ' +str(len(valTexts_Fr)))
#   
# y_val_Fr=[]
# with open('/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-test-Fr.cat','r') as f:
#     for line in f:
#         if line.strip() == "pos":
#             y_val_Fr.append(1)
#         else:
#             y_val_Fr.append(0)
# print ('The size of validation labels is: ' + str(len(y_val_Fr)))
# y_val_Fr = numpy.array(y_val_Fr)
#   
# # Build an indexed sequence for each repument
# vocabSize_Fr = 20000
#   
# tokenizer_Fr = text.Tokenizer(num_words=vocabSize_Fr,
#                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#                    lower=True,
#                    split=" ",
#                    char_level=False)
#   
# # Build the word to int mapping based on the training data
# tokenizer_Fr.fit_on_texts(trainTexts_Fr)
#   
# # Build the sequence (Keras built-in) for LSTM
# trainTextsSeq_Fr = tokenizer_Fr.texts_to_sequences(trainTexts_Fr)
# print (trainTextsSeq_Fr[0])
# print (len(trainTextsSeq_Fr))
# trainTextsSeq_Fr = numpy.array(trainTextsSeq_Fr)
#  
#   
# valTextsSeq_Fr = tokenizer_Fr.texts_to_sequences(valTexts_Fr)
# print (valTextsSeq_Fr[0])
# print (len(valTextsSeq_Fr))
# valTextsSeq_Fr = numpy.array(valTextsSeq_Fr)
#   
# # for non-sequence vectorization such as tfidf --> SVM
# trainVecMatrix_Fr = tokenizer_Fr.sequences_to_matrix(trainTextsSeq_Fr, mode='tfidf')
# #print (trainVecMatrix)
# print ('training vector length: '+str(len(trainVecMatrix_Fr)))
# print ('training vector columns: '+str(len(trainVecMatrix_Fr[0])))
#   
# valVecMatrix_Fr = tokenizer_Fr.sequences_to_matrix(valTextsSeq_Fr, mode='tfidf')
# print ('validation vector length: '+str(len(valVecMatrix_Fr)))
# print ('validation vector columns: '+str(len(valVecMatrix_Fr[0])))
#  
# dump_svmlight_file(trainVecMatrix_Fr, y_train_Fr, "trainVecMatrix_fr_s", zero_based=True, multilabel=False)
#  
# dump_svmlight_file(valVecMatrix_Fr, y_val_Fr, "valVecMatrix_fr_s", zero_based=True, multilabel=False)
