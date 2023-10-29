import torch, glob, cv2
import pandas as pd
import numpy as np
from pandas import DataFrame
torch.cuda.empty_cache()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

complexs = pd.read_csv('Validation/Validation_complex.csv',sep=',', header=0)
complexs = np.array(complexs)
complexs = torch.tensor(complexs)
name = pd.read_csv('Validation/drug.csv',sep=',')['SMILES']
name = name.values.tolist()
model=torch.load('6826/finish.pt')
positives = []
negatives = []

for complex in complexs:
    complex = complex.cuda()
    complex = complex.unsqueeze(0)
    complex = complex.unsqueeze(1)
    with torch.no_grad():
        output = model(complex)
        positive = output[0][1].data.cpu().numpy()
    positives.append(positive)
dic = {"SMILES": name, "predicts": positives}
data = DataFrame(dic)
data.to_csv("Validation/6826-3CL.csv",index=False)
print("predict is finished!")
