# Screen_Covalent_Compound_by_LISNN
# Title: Discovery of Covalent Lead Compounds Targeting 3CL Protease with Lateral Interactions Spiking Neural Network

## Introduction

Apply Lateral Interactions Spiking Neural Network to screening the covalent Lead Compounds targeting 3CL Pro.

## Requirements:

    Python        3.7
    torch-gpu     1.11.0
    numpy         1.21.5
    jieba         0.42.1
    pandas        1.3.5
    scikit-learn  1.0.2
    scipy         1.7.3
    seaborn       0.12.2
    torchaudio    0.11.0
    torchvision   0.12.0
    gensim        3.8.3
    matplotlib    3.5.3
    seaborn       0.12.2
    Other packages and their versions are shown in list.txt

## File description

    list.txt                      Packages and their versions required to run the environment
    
    compare_protein.xlsx          Protein amino acid sequence represented by numbers in figure S3
    
    3CL.csv                       Raw data on inhibitors targeting 3CL Pro
    
    model_human                   The file that evaluates the classification performance of the model
    
    
    model_Compounds_Inhibitory_Activity_Dataset_Targeting_3CL_Pro       The file of train model and screen inhibitors target 3CL pro
    
   model_Covalent_Complex_Dataset_Targeting_Cys                  The file of training and validating model and screen covalent compound targeting Cys
    
    
    ../LISNN.py                      The model of LISNN
    
    ../pre_data_embedding_data_interaction.py    ../pre_data_embedding.py   ../predict_embedding.py   ../Validation_predict_embedding.py               SMILES sequences of compounds and amino acid sequences of proteins are converted into vectors by Word2Vec.
    
    ../train.py                      Training models are based on different datasets
    
    ../predict.py   ../Validation_predict.py                    The probability of predicting the positive result
    
    ../model_Covalent_Complex_Dataset_Targeting_Cys/T-SNE.py    The model of t-SNE
    
    ../../data                          Relevant data to bulid the model
    
    ../../data/gensim-model-...         Word2Vec model

    ../../screen                        Relevant data for application model screening
   
    ../../seed                          The trained model represented by different seed number
    
    ../model_Covalent_Complex_Dataset_Targeting_Cys/screen/screen_specs           Processed commercial screening library
