import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM_model(nn.Module):
    def __init__(self, args):
        super(LSTM_model,self).__init__()

        self.args = args

        self.embedding_dim = args.embedding_dim
        self.input_dim = args.input_dim
        self.num_layers = args.num_layers
        self.rnn_dim = args.rnn_dim
        self.output_dim = args.output_dim
        self.seq_len = args.obs_len + args.pred_len
        self.use_gpu = args.use_gpu
        self.infer = args.infer

        self.input_layer = nn.Linear(self.input_dim, self.embedding_dim)
        self.lstm_layer = nn.LSTM(2, self.rnn_dim, self.num_layers)
        self.output_layer = nn.Linear(self.rnn_dim, self.output_dim)


    def forward(self, obs_traj, num_ped, pred_traj_gt):
        output, (hn, cn) = self.lstm_layer(obs_traj)

        pred_traj = Variable(torch.zeros(12, num_ped, 2)).cuda()  # tode
        for i in range(12):
            if self.infer:
                traj_tmp = self.output_layer(hn)
                pred_traj[i] = traj_tmp
                output, (hn, cn) = self.lstm_layer(traj_tmp, (hn, cn))
            else:
                traj_tmp = self.output_layer(hn)
                pred_traj[i] = traj_tmp
                output, (hn, cn) = self.lstm_layer(pred_traj_gt[i].view(1, num_ped, 2), (hn, cn))
        return pred_traj
