import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import maps
import pdb

from models_pytorch import *

def f_mdl_LA(x,c):
    D,_ = c.shape
    X = poly_kernel_matrix( [x],D-1 )
    # np.dot(poly_kernel_matrix( [x], c.shape[0]-1 ),c)
    return np.dot(X,c)

def poly_kernel_matrix( x,D ):
    '''
    x = single rela number data value
    D = largest degree of monomial

    maps x to a kernel with each row being monomials of up to degree=D.
    [1, x^1, ..., x^D]
    '''
    N = len(x)
    Kern = np.zeros( (N,D+1) )
    for n in range(N):
        for d in range(D+1):
            Kern[n,d] = x[n]**d;
    return Kern

def get_RLS_soln( X,Y,lambda_rls):
    N,D = X.shape
    XX_lI = np.dot(X.transpose(),X) + lambda_rls*N*np.identity(D)
    w = np.dot( np.dot( np.linalg.inv(XX_lI), X.transpose() ), Y)
    return w

def index_batch(X,batch_indices,dtype):
    if len(X.shape) == 1: # i.e. dimension (M,) just a vector
        batch_xs = torch.FloatTensor(X[batch_indices]).type(dtype)
    else:
        batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    return batch_xs

def get_batch2(X,Y,M,dtype):
    # TODO fix and make it nicer
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = index_batch(X,batch_indices,dtype)
    batch_ys = index_batch(Y,batch_indices,dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def main(argv=None):
    start_time = time.time()
    debug = True
    ##
    np.set_printoptions(suppress=True)
    ## true facts of the data set
    N = 5
    ## mdl degree and D
    Degree_mdl = 4
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    ## sgd params
    M = 5
    eta = 0.02 # eta = 1e-6
    nb_iter = int(100000)
    ## RLS params
    lambda_rls = 0.001
    #### Get Data set
    ## Get input variables X
    lb, ub = 0, 1
    x_true = np.linspace(lb,ub,N) # the real data points
    ## Get target variables Y
    #Y = np.sin(2*np.pi*x_true)
    Y = np.array([0.0,1.0,0.0,-1.0,0.0])
    Y.shape = (N,1)
    #
    X = poly_kernel_matrix(x_true,Degree_mdl)
    c_pinv = np.dot(np.linalg.pinv( X ),Y) # [D_pinv,1]
    c_rls = get_RLS_soln(X,Y,lambda_rls) # [D_pinv,1]
    ## data to TORCH
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    X = poly_kernel_matrix(x_true,Degree_mdl) # maps to the feature space of the model
    X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
    Y = Variable(torch.FloatTensor(Y).type(dtype), requires_grad=False)
    w_init=torch.randn(D_sgd,1).type(dtype)
    W = Variable( w_init, requires_grad=True)
    nb_module_params = 1
    #### Get models
    ## SGD model
    #mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits,b_inits=b_inits,bias=bias)
    #mdl_sgd = torch.nn.Sequential( torch.nn.Linear(D_sgd,1,bias=False) )
    #mdl_sgd = regression_NN(w_init=torch.randn(D_sgd,1).type(dtype))
    print('>>norm(Y): ', ((1/N)*torch.norm(Y)**2).data.numpy()[0] )
    #print('>>l2_loss_torch: ', (1/N)*( Y - mdl_sgd.forward(X)).pow(2).sum().data.numpy()[0] )
    #
    #nb_module_params = len( list(mdl_sgd.parameters()) )
    loss_list = [ ]
    #grad_list = [ [] for i in range(nb_module_params) ]
    for i in range(nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X,Y,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        #y_pred = mdl_sgd.forward(X)
        y_pred = batch_xs.mm(W)
        ## LOSS
        loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        W.data = W.data - eta*W.grad.data
        ## TRAINING STATS
        if i % 100 == 0 or i == 0:
            current_loss = loss.data.numpy()[0]
            loss_list.append(current_loss)
            if not np.isfinite(current_loss) or np.isinf(current_loss) or np.isnan(current_loss):
                print('loss: {} \n >>>>> BREAK HAPPENED'.format(current_loss) )
                break
        ## Manually zero the gradients after updating weights
        #mdl_sgd.zero_grad()
        W.grad.data.zero_()
    ##
    print('\a')
    #
    X, Y = X.data.numpy(), Y.data.numpy()
    #
    c_sgd = W.data.numpy()
    if debug:
        print('X = {} \n Y = {}'.format(X,Y))
        #print(mdl_sgd)
        print('c_sgd = ', c_sgd)
        print('c_pinv: ', c_pinv)
    #
    print('\n---- Learning params')
    print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {}'.format(Degree_mdl,N,M,eta,nb_iter))
    print('number of layers = {}'.format(nb_module_params))
    #
    print(' J(c_sgd) = ', (1/N)*(np.linalg.norm(np.dot( poly_kernel_matrix( x_true,c_sgd.shape[0]-1 ),c_sgd) - Y ))**2 )
    print( ' J(c_pinv) = ',(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_pinv))**2) )
    print( ' J(c_rls) = ',(1/N)*(np.linalg.norm(Y-(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_rls))**2) )**2) )
    ## plots
    x_horizontal = np.linspace(lb,ub,1000)
    X_plot = poly_kernel_matrix(x_horizontal,D_sgd-1)
    #plots objs
    #f_sgd = lambda x: f_mdl_LA(x,c_sgd)
    f_sgd = lambda x: np.dot(poly_kernel_matrix( [x], c_sgd.shape[0]-1 ),c_sgd)
    p_sgd, = plt.plot(x_horizontal, [ float(f_sgd(x_i)[0]) for x_i in x_horizontal ])
    p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
    p_data, = plt.plot(x_true,Y,'ro')
    p_list = [p_sgd,p_pinv,p_data]
    #
    plt.legend(p_list,['sgd curve Degree_mdl={}, batch-size= {}, iterations={}, eta={}'.format(
    str(D_sgd-1),M,nb_iter,eta),
    'min norm (pinv) Degree_mdl='+str(D_pinv-1),
    'data points'])
    plt.ylabel('f(x)')
    ##
    fig1 = plt.figure()
    p_loss, = plt.plot(np.arange(len(loss_list)), loss_list,color='m')
    plt.legend([p_loss],['plot loss'])
    plt.title('Loss vs Iterations')
    #
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    plt.show()

if __name__ == '__main__':
    main()
    print('\a')
