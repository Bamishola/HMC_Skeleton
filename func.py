import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

tolerance = 1e-8
fontS  = 12
colors = ['r', 'g', 'b', 'y', 'm']

def getSteadyState(P):
    theEigenvalues, leftEigenvectors = la.eig(P, right=False, left=True)
    theEigenvalues   = theEigenvalues.real
    leftEigenvectors = leftEigenvectors.real
    mask = abs(theEigenvalues - 1) < tolerance
    theEigenvalues   = theEigenvalues[mask]
    leftEigenvectors = leftEigenvectors[:, mask]
    attractorDistributions = leftEigenvectors / leftEigenvectors.sum(axis=0, keepdims=True)
    attractorDistributions = attractorDistributions.T
    theSteadyStates = np.sum(attractorDistributions, axis=0)
    return theSteadyStates

def getAlpha(Y, mu, var, I, t):
    N = np.size(Y)
    K = np.shape(mu)[0]
    alpha = np.zeros(shape=(N, K))
    S     = np.zeros(shape=(N))
    np1=0
    for k in range(K):
        alpha[np1, k] = I[k] * norm.pdf(Y[np1], loc=mu[k], scale=np.sqrt(var[k]))
    alpha[np1, :] /= np.sum(alpha[np1, :])
    for np1 in range(1, N):
        for k in range(K):
            for l in range(K):
                alpha[np1, k] += t[l, k] * alpha[np1-1, l]
            alpha[np1, k] *= norm.pdf(Y[np1], loc=mu[k], scale=np.sqrt(var[k]))
        S[np1] = np.sum(alpha[np1, :])
        alpha[np1, :] /= S[np1]
    return alpha, S


def getBeta(Y, mu, var, I, t, S):
    """
    Algorithme Backward.
    beta[n, k] = P(Y_{n+1},...,Y_{N-1} | X_n=k), normalisé.

    Initialisation : beta[N-1, k] = 1 pour tout k (pas d'observations futures).
    Récurrence (de n=N-2 à 0) :
        beta[n, k] = sum_l  t[k,l] * p(Y[n+1]|X[n+1]=l) * beta[n+1, l]
    Normalisation : on divise par S[n+1] (facteur calculé dans getAlpha)
    pour éviter les underflows numériques.
    """
    N = np.size(Y)
    K = np.shape(mu)[0]
    beta = np.zeros(shape=(N, K))

    # Initialisation au dernier instant
    beta[N-1, :] = 1.0

    # Récurrence backward
    for n in range(N-2, -1, -1):
        for k in range(K):
            for l in range(K):
                beta[n, k] += t[k, l] * norm.pdf(Y[n+1], loc=mu[l], scale=np.sqrt(var[l])) * beta[n+1, l]
        if S[n+1] > 0:
            beta[n, :] /= S[n+1]

    return beta


def getGamma(alpha, beta):
    """
    Probabilités marginales a posteriori (critère MPM).
    gamma[n, k] = P(X_n=k | Y_0,...,Y_{N-1})

    On multiplie terme à terme alpha et beta, puis on normalise
    chaque ligne pour que sum_k gamma[n,k] = 1.
    """
    N, K = np.shape(alpha)
    gamma = np.zeros(shape=(N, K))
    for n in range(N):
        gamma[n, :] = alpha[n, :] * beta[n, :]
        s = np.sum(gamma[n, :])
        if s > 0:
            gamma[n, :] /= s
    return gamma


def getMPMClassif(gamma):
    N = np.shape(gamma)[0]
    X_MPM = np.zeros(shape=(N))
    for n in range(N):
        X_MPM[n] = np.argmax(gamma[n, :])
    return X_MPM


def getConfMat(K, X, X_MPM):
    N = np.shape(X_MPM)[0]
    ConfMatrix = np.zeros(shape=(K,K))
    ERbyClass  = np.zeros(shape=(K))
    ERGlobal   = 0.
    for n in range(N):
        ConfMatrix[int(X[n]), int(X_MPM[n])] += 1.
        if X[n]!= X_MPM[n]: ERGlobal += 1.
    ERGlobal /= N
    for k in range(K):
        ERbyClass[k] = 1. - ConfMatrix[k, k] / np.sum(ConfMatrix, axis=1)[k]
    return ConfMatrix, ERGlobal, ERbyClass


def getProbaMarkov(JProba):
    K = np.shape(JProba)[0]
    IProba = np.sum(JProba, axis=1).T
    TProba = np.zeros(shape = np.shape(JProba))
    for r in range(K):
        TProba[r, :] = JProba[r, :] / IProba[r]
    return TProba, IProba


def InitParam(K, Y):
    mu    = np.zeros(shape=(K))
    sigma = np.zeros(shape=(K))
    c     = np.zeros(shape=(K, K))
    minY  = int(np.min(Y))
    maxY  = int(np.max(Y))
    varY  = np.var(Y)
    for k in range(K):
        sigma[k] = np.sqrt(varY / 2.)
        mu[k]    = minY + (maxY-minY)/(2.*K) + k * (maxY-minY)/K
        c[k, k]  = 0.9/K
        for l in range(k+1, K):
            c[k, l] = 0.1/(K*(K-1))
            c[l, k] = c[k, l]
    return mu, sigma, c


def getCtilde(Y, alpha, beta, tTabIter, meanTabIter, varTabIter, S):
    N = np.shape(Y)[0]
    K = np.shape(meanTabIter)[0]
    ctilde = np.zeros(shape=(N, K, K))
    for n in range(N-1):
        for xn in range(K):
            for xnp1 in range(K):
                ctilde[n, xn, xnp1] = tTabIter[xn, xnp1] * norm.pdf(Y[n+1], loc=meanTabIter[xnp1], scale=np.sqrt(varTabIter[xnp1])) * alpha[n, xn] * beta[n+1, xnp1] / S[n+1]
    return ctilde


def UpdateParameters(Y, gammatilde, ctilde):
    """
    Étape M de l'algorithme EM pour le modèle HMC.

    Mise à jour des paramètres par maximisation de l'espérance
    de la log-vraisemblance complète :

    mu[k]    = sum_n( gamma[n,k]*Y[n] ) / sum_n( gamma[n,k] )
    sigma[k] = sum_n( gamma[n,k]*(Y[n]-mu[k])^2 ) / sum_n( gamma[n,k] )
    c[k,l]   = sum_n( ctilde[n,k,l] )   (loi jointe)
    t[k,l]   = c[k,l] / sum_l(c[k,l])  (matrice de transition)
    I[k]     = gamma[0, k]              (loi initiale)
    """
    N, K = np.shape(ctilde)[0:2]

    mean  = np.zeros(shape=(K))
    sigma = np.zeros(shape=(K))
    c     = np.zeros(shape=(K, K))
    t     = np.zeros(shape=(K, K))
    I     = np.zeros(shape=(K))

    for k in range(K):
        denom   = np.sum(gammatilde[:, k])
        mean[k]  = np.sum(gammatilde[:, k] * Y) / denom
        sigma[k] = np.sum(gammatilde[:, k] * (Y - mean[k])**2) / denom

    # Loi jointe : somme de ctilde sur tous les instants n
    c = np.sum(ctilde, axis=0)

    # Matrice de transition et loi stationnaire (getProbaMarkov normalise c)
    t, _ = getProbaMarkov(c)

    # Loi initiale : marginale au premier instant
    I = gammatilde[0, :]

    return mean, sigma, c, t, I


def EM_Iter(iteration, Y, meanTabIter, varTabIter, cTabIter, tTabIter, ITabIter):
    N = np.shape(Y)[0]
    K = np.shape(meanTabIter)[1]
    alpha, S = getAlpha(Y, meanTabIter[iteration-1, :], varTabIter[iteration-1, :], ITabIter[iteration-1, :], tTabIter[iteration-1, :, :])
    beta     = getBeta (Y, meanTabIter[iteration-1, :], varTabIter[iteration-1, :], ITabIter[iteration-1, :], tTabIter[iteration-1, :, :], S)
    gamma    = getGamma(alpha, beta)
    ctilde   = getCtilde(Y, alpha, beta, tTabIter[iteration-1, :], meanTabIter[iteration-1, :], varTabIter[iteration-1, :], S)
    meanTabIter[iteration, :], varTabIter[iteration, :], \
        cTabIter[iteration, :, :], tTabIter[iteration, :, :], ITabIter[iteration, :] \
             = UpdateParameters(Y, gamma, ctilde)
    return gamma


def DrawCurvesParam(nbIter, pathToSave, meanTabIter, varTabIter, tTabIter):
    K = np.shape(meanTabIter)[1]
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    for k in range(K):
        ax1.plot(range(nbIter), meanTabIter[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
        ax2.plot(range(nbIter), varTabIter [:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
        ax3.plot(range(nbIter), tTabIter  [:, k, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
    ax1.set_ylabel('mu',       fontsize=fontS)
    ax2.set_ylabel('sigma**2', fontsize=fontS)
    ax3.set_ylabel('t(k,k)',   fontsize=fontS)
    ax1.legend()
    plt.xlabel('EM iterations', fontsize=fontS)
    plt.savefig(pathToSave + '_EvolParam.png', bbox_inches='tight', dpi=150)
    plt.close()


def DrawCurvesError(nbIter, pathToSave, MeanErrorRateTabbyClass, MeanErrorRateTab):
    K = np.shape(MeanErrorRateTabbyClass)[1]
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    for k in range(K):
        ax1.plot(range(nbIter), MeanErrorRateTabbyClass[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
    ax2.plot(range(nbIter), MeanErrorRateTab, lw=1, alpha=0.9, color='k', label='global')
    ax1.set_ylabel('% error', fontsize=fontS)
    ax2.set_ylabel('% error', fontsize=fontS)
    ax1.legend()
    plt.xlabel('EM iterations', fontsize=fontS)
    plt.savefig(pathToSave + '_EvolError.png', bbox_inches='tight', dpi=150)
    plt.close()
