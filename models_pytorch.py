import torch
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

class regression_NN(torch.nn.Module):

    def __init__(self,w_init):
        """
        """
        super(type(self), self).__init__()
        # mdl
        #self.W = Variable(w_init, requires_grad=True)
        #self.W = torch.nn.Parameter( Variable(w_init, requires_grad=True) )
        #self.W = torch.nn.Parameter( w_init ) 
        self.W = torch.nn.Parameter( w_init,requires_grad=True )
        #self.mod_list = torch.nn.ModuleList([self.W])

    def forward(self, x):
        """
        """
        y_pred = x.mm(self.W)
        return y_pred
