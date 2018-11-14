#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:05:19 2018

@author: lcampbe3

biocomputing exercise 10
"""

#1
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

data=pandas.read_csv('data.txt', header=0, sep=",")
d_f=pandas.DataFrame({'x':data.x, 'y':data.y})

#linear
def nllike(p,obs): #df is obs 
    B0=p[0] #placebo
    B1=p[1] #treatment
    sigma=p[2] #error
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

#quadratic
def quadlike(p, obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*obs.x*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

initialGuess_line=numpy.array([1,1,1])
initialGuess_quad=numpy.array([1,1,1,1])

fit_line=minimize(nllike,initialGuess_line,method="Nelder-Mead", options={'disp':True}, args=d_f)
fit_quad=minimize(quadlike,initialGuess_quad,method="Nelder-Mead", options={'disp':True}, args=d_f)

print(fit_line.x)
print(fit_quad.x)

from scipy import stats
teststat=2*(fit_line.fun-fit_quad.fun)
df=len(fit_quad.x)-len(fit_line.x)
1-stats.chi2.cdf(teststat,df) #p-value
#since the p-value is greater than 0.05, there 
#is no significant difference between the two models so use the simpler one

#2 ###################
import scipy
import scipy.integrate as spint

def ddSim(y, t0, r1, r2, a11, a12, a21, a22):
    N1=y[0]
    N2=y[1]
    dN1dt=r1*(1-N1*a11-N2*a12)*N1 #two differentials
    dN2dt=r2*(1-N2*a22-N1*a21)*N2
    return [dN1dt, dN2dt]



#model simulation 1 a12<a11 & a21<a22
params=(0.5, 0.5, 0.01, 0.005, 0.004, 0.025)
N0=[1,1]
times=range(0,100)

modelSim=spint.odeint(func=ddSim, y0=N0, t=times, args=params)
modelOutput=pandas.DataFrame({"t":times, "N1": modelSim[:,0],"N2":modelSim[:,1]})
ggplot(modelOutput)+geom_line(aes(x="t",y="N1"), color="blue")+geom_line(aes(x="t", y="N2"), color="green")

#model simulation 2 a12>a11 & a21<a22
params_second=(0.5, 0.5, 0.01, 0.2, 0.004, 0.025)
modelSim=spint.odeint(func=ddSim, y0=N0, t=times, args=params_second)
modelOutput=pandas.DataFrame({"t":times, "N1": modelSim[:,0],"N2":modelSim[:,1]})
ggplot(modelOutput)+geom_line(aes(x="t",y="N1"), color="blue")+geom_line(aes(x="t", y="N2"), color="green")

#model simulation 3 a12<a11 & a21>a22
params_third=(0.5, 0.5, 0.01, 0.2, 0.04, 0.025)
modelSim=spint.odeint(func=ddSim, y0=N0, t=times, args=params_third)
modelOutput=pandas.DataFrame({"t":times, "N1": modelSim[:,0],"N2":modelSim[:,1]})
ggplot(modelOutput)+geom_line(aes(x="t",y="N1"), color="blue")+geom_line(aes(x="t", y="N2"), color="green")














