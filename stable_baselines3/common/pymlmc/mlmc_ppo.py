from ast import arg
from datetime import datetime
from pickletools import read_uint1
import numpy
from math import sqrt
import sys
import warnings

from stable_baselines3.common.pymlmc.mlmc_fn import mlmc_fn

from .mlmc import mlmc

def mlmc_ppo(mlmc_fn, N, L, Eps, *args, **kwargs):
    """
    Multilevel Monte Carlo test routine. Prints results to stdout and file.

    Inputs:
        mlmc_fn: the user low-level routine for level l estimator. Its interface is

          (sums, cost) = mlmc_fn(l, N, *args, **kwargs)

          Inputs:  l: level
                   N: number of samples
                   *args, **kwargs: optional additional user variables

          Outputs: sums[0]: sum(Y)
                   sums[1]: sum(Y**2)
                   sums[2]: sum(Y**3)
                   sums[3]: sum(Y**4)
                   sums[4]: sum(P_l)
                   sums[5]: sum(P_l**2)
                      where Y are iid samples with expected value
                          E[P_0]            on level 0
                          E[P_l - P_{l-1}]  on level l > 0
                   cost: user-defined computational cost of N samples

        N:    number of samples for convergence tests
        L:    number of levels for convergence tests

        Eps:  desired accuracy (rms error) array for MLMC calculations

        *args, **kwargs: optional additional user variables to be passed to mlmc_fn
    """

    del1 = []
    del2 = []
    var1 = []
    var2 = []
    kur1 = []
    chk1 = []
    cost = []

    suml = numpy.zeros((2, L+1))
    costl = numpy.zeros(L+1)

    for l in range(0, L+1):

        (sums, cst) = mlmc_fn(l, N, *args, **kwargs)
        suml[0, l]   = suml[0, l] + sums[0]
        suml[1, l]   = suml[1, l] + sums[1]
        costl[l]     = costl[l] + cst
        sums = sums/N
        cst = cst/N

        if l == 0:
            kurt = 0.0
        else:
            kurt = (     sums[3]
                     - 4*sums[2]*sums[0]
                     + 6*sums[1]*sums[0]**2
                     - 3*sums[0]*sums[0]**3 ) / (sums[1]-sums[0]**2)**2

        cost.append(cst)
        del1.append(sums[0])
        del2.append(sums[4])
        var1.append(sums[1]-sums[0]**2)
        var2.append(max(sums[5]-sums[4]**2, 1.0e-10)) # fix for cases with var = 0
        kur1.append(kurt)

        if l == 0:
            check = 0
        else:
            check =          abs(       del1[l]  +      del2[l-1]  -      del2[l])
            check = check / ( 3.0*(sqrt(var1[l]) + sqrt(var2[l-1]) + sqrt(var2[l]) )/sqrt(N))
        chk1.append(check)

    if kur1[-1] > 100.0:
        warnings.warn("\n WARNING: kurtosis on finest level = %f \n" % kur1[-1])
        warnings.warn(" indicates MLMC correction dominated by a few rare paths; \n")
        warnings.warn(" for information on the connection to variance of sample variances,\n")
        warnings.warn(" see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n\n")

    if max(chk1) > 1.0:
        warnings.warn("\n WARNING: maximum consistency error = %f \n" % max(chk1))
        warnings.warn(" indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied; \n")
        warnings.warn(" to be more certain, re-run mlmc_test with larger N \n\n")

    # Use linear regression to estimate alpha, beta and gamma
    L2 = L+1
    L1 = max(0, L2-3)
    pa    = numpy.polyfit(range(L1+1, L2+1), numpy.log2(numpy.abs(del1[L1:L2])), 1);  alpha = -pa[0]
    pb    = numpy.polyfit(range(L1+1, L2+1), numpy.log2(numpy.abs(var1[L1:L2])), 1);  beta  = -pb[0]
    pg    = numpy.polyfit(range(L1+1, L2+1), numpy.log2(numpy.abs(cost[L1:L2])), 1);  gamma =  pg[0]

    # Second, MLMC complexity tests

    # alpha = max(alpha, 0.5)
    # beta  = max(beta, 0.5)
    theta = 0.5

    ml = numpy.abs(       suml[0, :]/N)
    Vl = numpy.maximum(0, suml[1, :]/N - ml**2)
    Cl = costl/N

    P_ml, N_ml, C_ml, V_ml = [],[],[],[]
    P_mc, N_mc = [],[]
        
    V_L = var1[0]
    C_l = []
    for i,_ in enumerate(var2):
        C_array = numpy.array([-1,1]*(i+1))[-(i+1):]*Cl[:(i+1)]
        C_l.append(numpy.sum(C_array))

    for eps in Eps:
        P_ml_, Nl = mlmc_estimates(mlmc_fn, Vl, Cl, eps, theta, alpha)
        
        P_ml.append( round(P_ml_,4) )
        N_ml.append( [ int(elem) for elem in Nl ] )
        C_ml.append( [ round(elem, 2) for elem in Cl ] )
        V_ml.append( [round(elem, 2) for elem in Vl])
        
        mc_cost  = (V_L*Cl[-1])/((1.0 -theta)*eps**2)
        N_mc_ = numpy.ceil( mc_cost/C_l[-1]).astype(int)
        P_mc_ = mc_estimates(mlmc_fn, N_mc_, L)

        P_mc.append( round(P_mc_,4) )
        N_mc.append( N_mc_ )
        
    expt_results = {'N':N, 'C_l':[round(c,2) for c in C_l], 'V_l':[round(v,2) for v in var2], 'P_l':[round(p,4) for p in del2]}
    mc_results = {'eps_mc':Eps, 'P_mc':P_mc, 'N_mc':N_mc, 'C_mc':round(C_l[-1],2), 'V_mc':round(V_L,2)}
    ml_results = {'eps_ml':Eps, 'P_ml':P_ml, 'N_ml':N_ml, 'C_ml':C_ml, 'V_ml':V_ml, 'a,b,g':[round(alpha,2), round(beta,2), round(gamma,2)]}

    return expt_results, mc_results, ml_results

def mc_estimates(mlmc_fn, N, L, *args, **kwargs):
    (sums, _) = mlmc_fn(L, N, *args, **kwargs)
    return sums[4]/N


def mlmc_estimates(mlmc_fn, Vl, Cl, eps, theta, alpha, *args, **kwargs):

    L = Cl.shape[0]-1
    suml = numpy.zeros((2, L+1))

    Nl = numpy.ceil( numpy.sqrt(Vl/Cl) * sum(numpy.sqrt(Vl*Cl)) / ((1-theta)*eps**2) )
    for l,N in zip(range(0, L+1),Nl):
        (sums, _) = mlmc_fn(l, int(N), *args, **kwargs)
        suml[0, l]   = suml[0, l] + sums[0]
        suml[1, l]   = suml[1, l] + sums[1]

    # check weak convergence
    ml = numpy.abs(       suml[0, :]/Nl)
    rang = list(range(min(3, L)))
    rem = ( numpy.amax(ml[[L-x for x in rang]] / 2.0**(numpy.array(rang)*alpha))
                / (2.0**alpha - 1.0) )
    if rem > numpy.sqrt(theta)*eps:
        warnings.warn(f"Weak convergence not met, remaning error ({rem:.2f}) should be smaller than {numpy.sqrt(theta)*eps:.2f} ")

    return sum(suml[0,:]/Nl), Nl


