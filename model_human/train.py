import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
import random
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,roc_curve,auc
from LISNN import LISNN

parser = argparse.ArgumentParser(description='train_data_interaction.py')

parser.add_argument('-gpu', type = int, default = 0)
parser.add_argument('-epoch', type = int, default = 200)
parser.add_argument('-batch_size', type = int, default = 256)
parser.add_argument('-learning_rate', type = float, default = 0.002)
parser.add_argument('-if_lateral', type = bool, default = True)

seed=random.randint(1000,10000)
print(seed)
opt = parser.parse_args()
test_scores = []
train_scores = []

save_path = str(seed)
if not os.path.exists(save_path):
    os.mkdir(save_path)

torch.cuda.set_device(opt.gpu)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


Y=pd.read_csv('data/data_interaction_shuffle.csv',sep=',')['interactions']
Y=np.array(Y)
Y = torch.LongTensor(Y)
X=pd.read_csv('data/BindingDBall-100dim-complex.csv',sep=',', header=0)
X=np.array(X)
X = torch.Tensor(X)

#train_set and test_set
my_dataset = data.TensorDataset(X, Y)
train_size = len(my_dataset)*0.8
train_size = int(train_size)
test_size = len(my_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])

#input
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = opt.batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = opt.batch_size, drop_last=True)

#model
model = LISNN(opt)
model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 75, gamma = 0.1)

#train
def train(epoch):
    model.train()
    scheduler.step()
    correct = 0
    total = 0
    total_loss = 0
    for i, (complex, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        complex = Variable(complex.cuda())
        # complex = Variable(complex)
        complex = complex.unsqueeze(1)
        labels=torch.tensor(labels)
        one_hot = torch.zeros(opt.batch_size, 2).scatter(1, labels.unsqueeze(1), 1)
        labels = Variable(one_hot.cuda())
        # labels = Variable(one_hot)
        outputs = model(complex)
        loss = loss_function(outputs, labels)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
    print('Epoch: %d, Loss: %.4f' % (epoch + 1, total_loss))
    total_loss = 0

#test
def eval(epoch, if_test):
    model.eval()
    correct = 0
    total = 0
    score_train=[]
    label_train=[]
    score_test = []
    label_test = []
    out_put=[]
    bins=([0,0.5,1])
    if if_test:
        for i, (com_plex, labels) in enumerate(test_loader):
            com_plex = com_plex.cuda()
            com_plex = com_plex.unsqueeze(1)
            labels = labels.cuda()
            outputs= model(com_plex)
            pred_output=torch.softmax(outputs,1).cpu().numpy()
            out_put.extend(pred_output.tolist())
            pre_lables = torch.max(outputs, 1)[1].data.cpu().numpy().tolist()
            labels = labels.data.cpu().numpy().tolist()
            score_test.extend(pre_lables)
            label_test.extend(labels)
        out_put= pd.DataFrame(out_put)
        out_put.to_csv("out_put.csv")
        out_put_proba = list(pd.read_csv("out_put.csv")["1"])
        os.remove("out_put.csv")
        acc = accuracy_score(label_test, score_test)
        precision = precision_score(label_test, score_test, average='weighted')
        test_scores.append(acc)
        tn,fp,fn,tp=np.histogram2d(label_test, score_test,bins=bins)[0].flatten()
        SP=tn/(tn+fp)
        SE=tp/(tp+fn)
        fpr, tpr, thresholds_keras = roc_curve(label_test, out_put_proba)
        auc_score = auc(fpr, tpr)
        plt.figure()
        plt.plot([0, 1],[0, 1], 'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        if acc >=max(test_scores):
            save_file = "finish"+ '.pt'
            torch.save(model, os.path.join(save_path, save_file))
            plt.savefig('%s/roc.jpg' %(save_path))
            plt.cla()
            plt.close()
            metrics ={'Epoch': ('%d' % (epoch+1)),'Accuracy': ('%.2f%%' % (acc*100)), 'Precision': ('%.2f%%' % (precision*100)),
                      'SP': ('%.2f%%' % (SP*100)), 'SE': ('%.2f%%' % (SE*100))}
            metrics = pd.DataFrame(metrics, index=[0])
            metrics.to_csv('%s/metrics.csv'%(save_path))
        print('Test Accuracy: %.2f%%' % (acc*100))
        print('Precision:     %.2f%%' % (precision*100))
        print('SP:            %.2f%%' % (SP*100))
        print('SE:            %.2f%%' % (SE*100))
        print('AUC:           %.4f'   % (auc_score))

    else:
        for i, (com_plex, labels) in enumerate(train_loader):
            com_plex = com_plex.cuda()
            com_plex = com_plex.unsqueeze(1)
            labels = labels.cuda()
            outputs = model(com_plex)
            predicted = torch.max(outputs, 1)[1]
            labels = labels.data.cpu().numpy()
            lables = labels.tolist()
            predicted = predicted.data.cpu().numpy()
            predicted = predicted.tolist()
            score_train.extend(predicted)
            label_train.extend(labels)
        acc = accuracy_score(label_train, score_train)
        print('Train Accuracy: %.2f%%' % (acc*100))
        train_scores.append(acc)

#plot
def picture(x,y,save_path,z):
    plt.cla()
    plt.subplot(1,1,1)
    plt.plot(x,y)
    plt.title("%s accuracy" %(z))
    plt.ylabel("%s accuracy" %(z))
    plt.savefig("%s/%s accuracy.jpg" %(save_path,z))


def main():
    start_time = time.time()
    for epoch in range(opt.epoch):
        train(epoch)
        with torch.no_grad():
            eval(epoch, if_test = False)
            torch.cuda.empty_cache()
            eval(epoch, if_test = True)
    print(str(max(train_scores)*100)+"%")
    print(str(max(test_scores)*100)+"%")
    picture(range(0, epoch+1), train_scores, save_path, 'train')
    picture(range(0, epoch+1), test_scores, save_path, 'test')
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
