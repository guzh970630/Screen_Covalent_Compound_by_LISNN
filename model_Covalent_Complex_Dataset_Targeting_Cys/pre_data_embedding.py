import sys
import pandas as pd
import numpy as np
import gensim 
import jieba
from gensim.models import Word2Vec 
import random 
from sklearn.decomposition import IncrementalPCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import os 
import random
import seaborn as sns
from sklearn.utils import shuffle

os.mkdir('data')
data1= pd.read_csv('data1.csv',sep=',')
data1=shuffle(data1)
data1.to_csv('data/data1_shuffle.csv', sep=",")

protein_seq = pd.read_csv('data/data1_shuffle.csv',sep=',')['ProSeqs']
drug_Smi= pd.read_csv('data/data1_shuffle.csv',sep=',')['Smiles']

class SPVec:
    def protein2vec(dims, window_size, negative_size):
        word_vec = pd.DataFrame()
        dictionary = []
        Index = []
        texts = [[word for word in re.findall(r'.{3}', str(document))] for document in list(protein_seq)]
        print(texts)
        model = Word2Vec(texts, size=dims, window=window_size, min_count=1, negative=negative_size, sg=1, sample=0.001, hs=1, workers=4)
        model.save('data/gensim-model-100dim-protein')
        new_model = gensim.models.Word2Vec.load('data/gensim-model-100dim-protein')
        print(new_model)
        vectors = pd.DataFrame([new_model[word] for word in (new_model.wv.vocab)])
        vectors['Word'] = list(new_model.wv.vocab)
        print(vectors)

        for i in range(len(protein_seq)):
            Index.append(i)
        # Word segmentation
        for i in range(len(texts)):
            i_word = []
            for w in range(len(texts[i])):
                i_word.append(Index[i])
            dictionary.extend(i_word)
        word_vec['Id'] = dictionary

        # word vectors generation
        dictionary = []
        for i in range(len(texts)):
            i_word = []
            for w in range(len(texts[i])):
                i_word.append(texts[i][w])
            dictionary.extend(i_word)
        word_vec['Word'] = dictionary
        del dictionary, i_word
        word_vec = word_vec.merge(vectors, on='Word', how='left')
        word_vec.columns = ['Id'] + ['word'] + ["vec_{0}".format(i) for i in range(0, dims)]
        print(word_vec)
        return word_vec

    def smiles2vec(dims, window_size, negative_size):
        word_vec = pd.DataFrame()
        dictionary = []
        Index = []
        texts = [[word for word in re.findall(r'.{3}', str(document))] for document in list(drug_Smi)]
        print(texts)
        print(len(texts))
        model = Word2Vec(texts, size=dims, window=window_size, min_count=1, negative=negative_size, sg=1, sample=0.001, hs=1, workers=4)
        model.save('data/gensim-model-100dim-smiles')
        new_model = gensim.models.Word2Vec.load('data/gensim-model-100dim-smiles')
        print(new_model)
        vectors = pd.DataFrame([new_model[word] for word in (new_model.wv.vocab)])
        vectors['Word'] = list(new_model.wv.vocab)
        print(vectors)

        for i in range(len(drug_Smi)):
            Index.append(i)
        # Word segmentation
        for i in range(len(texts)):
            i_word = []
            for w in range(len(texts[i])):
                i_word.append(Index[i])
            dictionary.extend(i_word)
        word_vec['Id'] = dictionary

        # word vectors generation
        dictionary = []
        for i in range(len(texts)):
            i_word = []
            for w in range(len(texts[i])):
                i_word.append(texts[i][w])
            dictionary.extend(i_word)
        word_vec['Word'] = dictionary
        del dictionary, i_word
        word_vec = word_vec.merge(vectors, on='Word', how='left')
        word_vec.columns = ['Id'] + ['word'] + ["vec_{0}".format(i) for i in range(0, dims)]
        print(word_vec)
        return word_vec

    # Molecular Structure and Protein Sequence Representation
    def feature_embeddings_protein(dims):
        protein_vec = SPVec.protein2vec(100, 12, 15)
        protein_vec = protein_vec.drop('word', axis=1)
        name = ["vec_{0}".format(i) for i in range(0, dims)]
        feature_embeddings = pd.DataFrame(protein_vec.groupby(['Id'])[name].agg('mean')).reset_index()
        feature_embeddings.columns = ["Index"] + ["mean_ci_{0}".format(i) for i in range(0, dims)]
        return feature_embeddings

    def feature_embeddings_smiles(dims):
        smiles_vec = SPVec.smiles2vec(100, 12, 15)
        smiles_vec = smiles_vec.drop('word', axis=1)
        name = ["vec_{0}".format(i) for i in range(0, dims)]
        feature_embeddings = pd.DataFrame(smiles_vec.groupby(['Id'])[name].agg('mean')).reset_index()
        feature_embeddings.columns = ["Index"] + ["mean_ci_{0}".format(i) for i in range(0, dims)]
        return feature_embeddings



if __name__=='__main__':
    print ("Molecular Structure and Protein Sequence Continuous Representation")

    drug_vec=SPVec.smiles2vec(3, 12, 15)
    drug_embeddings = SPVec.feature_embeddings_smiles(100)
    drug_embeddings['smiles'] = drug_Smi
    drug_embeddings1 = drug_embeddings.drop(['Index', 'smiles'], axis=1)
    print(drug_embeddings1)
    drug_embeddings1.to_csv('data/100dim-drug.csv', index=False, sep=',')

    prot_vec = SPVec.protein2vec(3, 12, 15)
    prot_embeddings = SPVec.feature_embeddings_protein(100)
    prot_embeddings['proteinseq']=protein_seq
    prot_embeddings1= prot_embeddings.drop(['Index', 'proteinseq'], axis=1)
    print(prot_embeddings1)
    prot_embeddings1.to_csv('data/100dim-protein.csv', index=False, sep=',')
    print('******Molecular Structure and Protein Sequence Continuous Representation finished!*********')
    x_pro = pd.read_csv('data/100dim-protein.csv', sep=',', header=0)
    x_drug = pd.read_csv('data/100dim-drug.csv', sep=',', header=0)
    X = pd.concat([x_pro, x_drug], axis=1)
    X.to_csv('data/100dim-complex.csv', index=False, sep=',')
    print("finish")


