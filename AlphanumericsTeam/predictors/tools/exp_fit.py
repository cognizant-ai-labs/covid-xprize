import numpy as np


def get_exp_fit(new_cases, window_length, time_unit):

    L = len(new_cases)
    new_cases_log = np.log(new_cases)
    ALog = np.zeros(L)
    r = np.zeros(L)
    n = np.arange(-window_length, 0)
    En = np.mean(n)
    En2 = np.mean(np.power(n, 2))
    Det = En2 - En*En

    for x in range(window_length, L):
        segment = new_cases_log[x - window_length: x]
        ALog[x] = (np.mean(segment)*En2 - np.mean(n*segment)*En)/Det
        r[x] = (np.mean(n*segment) - np.mean(segment)*En)/Det


    Rt = np.exp(r)
    A = np.exp(ALog)
    Lambda = r/time_unit
    ExpFit = A*Rt
    return (Rt, A, Lambda, ExpFit)
