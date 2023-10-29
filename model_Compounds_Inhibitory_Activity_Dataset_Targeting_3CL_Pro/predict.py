import torch, glob, cv2
import pandas as pd
import numpy as np
from pandas import DataFrame
torch.cuda.empty_cache()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

complexs = pd.read_csv('screen/screen_specs_drug.csv',sep=',', header=0)
complexs = np.array(complexs)
complexs = torch.tensor(complexs)
name = pd.read_csv('screen/screen_specs.csv',sep=',')['name']
name = name.values.tolist()
model=torch.load('1936/finish.pt')
positives = []

for complex in complexs:
    complex = complex.cuda()
    complex = complex.unsqueeze(0)
    complex = complex.unsqueeze(1)
    with torch.no_grad():
        output = model(complex)
        positive = output[0][1].data.cpu().numpy()
    positives.append(positive)
dic = {"names": name, "predicts": positives}
data = DataFrame(dic)
data.to_csv("screen/1936.csv",index=False)
print("predict is finished!")
