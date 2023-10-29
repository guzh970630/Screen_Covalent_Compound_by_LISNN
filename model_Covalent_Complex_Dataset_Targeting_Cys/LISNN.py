import torch
import torch.nn as nn
import torch.nn.functional as F

thresh, lens, decay, if_bias = (0.5, 0.5, 0.5, True)

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

# lateral = None

def mem_update(ops, x, mem, spike, lateral = None):
    mem = mem * decay * (1. - spike) + ops(x)
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike

class LISNN(nn.Module):
    def __init__(self, opt):
        super(LISNN, self).__init__()
        self.batch_size = opt.batch_size
        self.fc = (512, 2)

        self.cnn = ((1, 64, 3, 1, 1, 5, 2), (64, 128, 3, 1, 1, 5, 2), (128, 128, 3, 1, 1, 5, 2))
        self.kernel = (200, 100, 50, 25)

        self.conv1 = nn.Conv1d(self.cnn[0][0], self.cnn[0][1], kernel_size = self.cnn[0][2], stride = self.cnn[0][3], padding = self.cnn[0][4], bias = if_bias)
        self.conv2 = nn.Conv1d(self.cnn[1][0], self.cnn[1][1], kernel_size = self.cnn[1][2], stride = self.cnn[1][3], padding = self.cnn[1][4], bias = if_bias)
        self.conv3 = nn.Conv1d(self.cnn[2][0], self.cnn[2][1], kernel_size = self.cnn[2][2], stride = self.cnn[2][3], padding = self.cnn[2][4], bias = if_bias)
        self.fc1 = nn.Linear(self.kernel[-1] * self.cnn[-1][1], self.fc[0], bias = if_bias)
        self.fc2 = nn.Linear(self.fc[0], self.fc[1], bias = if_bias)

        if opt.if_lateral:
            self.lateral1 = nn.Conv1d(self.cnn[0][1], self.cnn[0][1], kernel_size=self.cnn[0][5], stride=self.cnn[0][3],
                                      padding=self.cnn[0][6], groups=self.cnn[0][1], bias=False)
            self.lateral2 = nn.Conv1d(self.cnn[1][1], self.cnn[1][1], kernel_size=self.cnn[1][5], stride=self.cnn[1][3],
                                      padding=self.cnn[1][6], groups=self.cnn[1][1], bias=False)
            self.lateral3 = nn.Conv1d(self.cnn[2][1], self.cnn[2][1], kernel_size=self.cnn[2][5], stride=self.cnn[2][3],
                                      padding=self.cnn[2][6], groups=self.cnn[2][1], bias=False)
        else:
            self.lateral1 = None
            self.lateral2 = None
            self.lateral3 = None


    def forward(self, input, time_window = 10):
        c1_mem = c1_spike = torch.zeros(self.batch_size, self.cnn[0][1], self.kernel[0]).cuda()
        c2_mem = c2_spike = torch.zeros(self.batch_size, self.cnn[1][1], self.kernel[1]).cuda()
        c3_mem = c3_spike = torch.zeros(self.batch_size, self.cnn[2][1], self.kernel[2]).cuda()

        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.fc[0]).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.fc[1]).cuda()

        for step in range(time_window):

            c1_mem, c1_spike = mem_update(self.conv1, input.float(), c1_mem, c1_spike, self.lateral1)
            x = F.avg_pool1d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem,c2_spike, self.lateral2)
            x = F.avg_pool1d(c2_spike, 2)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike, self.lateral3)
            x = F.avg_pool1d(c3_spike, 2)

            x = x.view(self.batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs