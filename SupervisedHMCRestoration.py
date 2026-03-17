import numpy as np
import matplotlib.pyplot as plt
from func import *
from os.path import join

if __name__ == '__main__':

    reposource = './sources'
    reporesult = './results'

    #############################################@
    ####### Set the parameters and read the data

    # Gaussians
    mu  = [100, 110]
    var = [6**2, 3**2]
    K   = 2 # le nombre de classes

    # Homogeneous and stationary Markov chain
    # t = np.array([[0.7, 0.1, 0.2], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])
    # t = np.array([[0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.3, 0.5, 0.2]])
    t = np.array([[0.95, 0.05], [0.05, 0.95]])

    I = getSteadyState(t)
    print('I=', I, '\nt=', t)
    
    # The data (simulated)
    X, Y = np.loadtxt(join(reposource, 'XY.out'))
    N    = X.shape[0]

    #############################################@
    ####### MPM Restoration
    
    # forward computing
    alpha, S = getAlpha(Y, mu, var, I, t)
    
    # backward computing
    beta = getBeta(Y, mu, var, I, t, S)
    
    # gamma computing (marginal a posterori proba)
    gamma = getGamma(alpha, beta)

    # MPM classification
    X_MPM = getMPMClassif(gamma)

    # error rate compuation
    ConfMatrix_MPM, ERGlobal_MPM, ERbyClass_MPM = getConfMat(K, X, X_MPM)
    print('Confusion matrix for MPM:\n', ConfMatrix_MPM)
    print('Global error rate for MPM:', ERGlobal_MPM)
    print('By class error rate for MPM:', ERbyClass_MPM)
    
    #############################################@
    ####### MAP Restoration
    
    # # MAP classification (Viterbi algo)
    # X_MAP = getMAPClassif(Y, mu, var, I, t)
    
    # # error rate compuation
    # ConfMatrix_MAP, ERGlobal_MAP, ERbyClass_MAP = getConfMat(K, X, X_MAP)
    # print('Confusion matrix for MAP:\n', ConfMatrix_MAP)
    # print('Global error rate for MAP:', ERGlobal_MAP)
    # print('By class error rate for MAP:', ERbyClass_MAP)
    # 