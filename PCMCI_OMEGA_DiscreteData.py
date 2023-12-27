# FILENAME:  PCMCI_OMEGA_discrete.py
# Author: Shanyun Gao
# Paper: https://openreview.net/forum?id=dYeUvLUxBQ
# Note that this code utilizes the old verision of tigramite package(v5.1.0), with a significant portion aligning with the logic present in functions from the tigramite package.
import time
from cmath import sqrt
from pickle import TRUE
import numpy as np
from numpy import sum as sum
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm
import sklearn
import pandas as pd
import scipy
import math
import random
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy import nan
from math import isnan
from copy import deepcopy
from numpy.random.mtrand import random_integers
import pyAgrum as gum
import os
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import independence_tests
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction
# import os
# import sys

################################################Algorithm for Discrete Data#####################################
#Note:  1. function "algorithm_v2_mci_" is the main function for PCMCI_Omega with "turning point" strategy. 
#          The function inputs are "data", "\tau_{ub}" and "\omega_{ub}". "data" has shape [T,N] where T is the time length and N denotes the N-var time series.
#          The function output "tem_array" is the estimated edge array with shape [N,\Omega,N,tau_max_pcmci+1] where the explanation of \Omega can be found in paper.
#          With the same logic in tigramite package, if the target variable has index j, then the incoming edge array is [N,\Omega,j,tau_max_pcmci+1].
#          The function output "omega_hat4" is the estimated periodicity array for each time series.
#          The function output "superset_bool" is the super set of parents set obtained from PCMCI, denoted by "\widehat{SPA}" in paper.
#       2. This code is based on tigramite package version v5.1.0. Updates for a newer version of the tigramite package will be available at a later time.

def group_in_threes(slicable):
    for i in range(len(slicable)-2):
        yield slicable[i:i+3]

def turning_points(L):
    iloc=[]
    if L[0]<L[1]:
      iloc=[0]
    if len(np.where(L==next(x for x in L if not isnan(x)))[0])!=0:
      iloc.append(np.where(L==next(x for x in L if not isnan(x)))[0][0])
    for index, three in enumerate(group_in_threes(L)):
        if (three[0] > three[1] <= three[2]):
            #yield index + 1
            iloc.append(index+1)
    return iloc

def all_equal(iterator):
  iterator = iter(iterator)
  try:
      first = next(iterator)
  except StopIteration:
      return True
  return all(first == x for x in iterator)

def LCMofArray(a):
  lcm = a[0]
  for i in range(1,len(a)):
    lcm = lcm*a[i]//math.gcd(lcm, a[i])
  return lcm


def algorithm_v2_mci_(data,tau_max_pcmci,search_omega_bound):
  st = time.time()
  T=data.shape[0]
  N=data.shape[1]
  results_variable=np.zeros(shape=(N,search_omega_bound,search_omega_bound))
  results_variable[:][:][:] = nan
  num_edge1=np.zeros(shape=(N,search_omega_bound))
  num_edge1[:][:] = nan
  omega_hat1=np.zeros(shape=(N))
  num_edge2=np.zeros(shape=(N,search_omega_bound))
  num_edge2[:][:] = nan
  omega_hat2=np.zeros(shape=(N))
  num_edge3=np.zeros(shape=(N,search_omega_bound))
  num_edge3[:][:] = nan
  omega_hat3=np.zeros(shape=(N))
  num_edge4=np.zeros(shape=(N,search_omega_bound))
  num_edge4[:][:] = nan
  omega_hat4=np.zeros(shape=(N))
  turning_points_single_var=np.zeros(shape=(N,search_omega_bound))
  var_names = [0]*N
  for i in range(N):
    var_names[i] = r'$X^{}$'.format(i)
# var_names = [r'$X^0$', r'$X^1$', r'$X^2$']  
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))}, 
                            var_names=var_names)
  cmi_symb = CMIsymb(significance='shuffle_test')
  pcmci_cmi_symb = PCMCI(
      dataframe=dataframe, 
      cond_ind_test=cmi_symb,
      verbosity=1)
  pcmci_cmi_symb.verbosity = 1
  results = pcmci_cmi_symb.run_pcmci(tau_min=1,tau_max=tau_max_pcmci, pc_alpha=0.2, alpha_level=0.1)

  superset_bool=np.array(results["p_matrix"]<0.01, dtype=int)
  superset_dict_est = {}
  for i in range(N):
      superset_dict_est[i] = []
  for k in range(N):
    for i in range(N):
      for j in range(len(results["p_matrix"][0,0,:])):
        if superset_bool[i,k,j]==1:
          #  key='{}'.format(k)
          superset_dict_est[k].append((i,-j))

  #Superset_dict_est will be used in MCI test and then select the Omega_hat.
  for i in range(1,search_omega_bound+1):#search omega=1,2
    print("The current omega is {},total #Omega is {}".format(i,search_omega_bound))
    for j in range(0,i):
      print("The current sub_sample is {},total #sub_sample under omega= {} is {}".format(j,i,i))
      dataframe = pp.DataFrame(data_run, mask= mask_all[i-1][j][N:T,:],
                              datatime = {0:np.arange(len(data_run))}, 
                              var_names=var_names)
      cmi_symb = CMIsymb(significance='shuffle_test',mask_type='y')
      pcmci_cmi_symb = PCMCI(
      dataframe=dataframe, 
      cond_ind_test=cmi_symb,
      verbosity=1)
      pcmci_cmi_symb.verbosity = 1
      results = pcmci_cmi_symb.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci, parents=superset_dict_est, alpha_level=0.1) 

      for k in range(N):
        results_variable[k][i-1][j]=sum(sum(results["p_matrix"][:][k]<0.01))
        #print(results_variable)
  fix = deepcopy(results_variable)

  for i in range(1,search_omega_bound+1):#search omega=1,2
    for k in range(N):
      #num_edge[k][i-1]=np.sum(results_variable[k][i-1]) #i:omega
      num_edge1[k][i-1]=np.nanmean(results_variable[k][i-1])
      num_edge2[k][i-1]=np.nanmax(results_variable[k][i-1])
      num_edge3[k][i-1]=np.nanmin(results_variable[k][i-1])

  for k in range(N):
    print(k)
    omega_hat1[k]=np.where(num_edge1[k,:]==num_edge1[k,:].min())[0][0]+1
    omega_hat2[k]=np.where(num_edge2[k,:]==num_edge2[k,:].min())[0][0]+1
    omega_hat3[k]=np.where(num_edge3[k,:]==num_edge3[k,:].min())[0][0]+1
    temp_list=[]
    for j in range(0,search_omega_bound-1):
      temp_list.append(turning_points(results_variable[k][:,j]))
    print(temp_list)
    if len(list(set.intersection(*map(set, temp_list))))==0:
      omega_hat4[k]=nan
    else:
      turning_points_single_var[k]=list(set.intersection(*map(set, temp_list)))[0]
      omega_hat4[k]=np.amin(turning_points_single_var[k])+1
  for k in range(N):
    for i in range(1,search_omega_bound+1):#search omega=1,2
     for j in range(0,i):
        if results_variable[k][0][0]<=np.min((results_variable[k][1][0],results_variable[k][1][1])):
         omega_hat4[k]=1
  for k in range(N):
    if np.isnan(omega_hat4[k]):
      omega_hat4[k]=omega_hat2[k]
  #suppose omega_hat4=[2,2,1,1,1]
  print("Omega hat is {}".format(omega_hat4))
  print("Omega is {}".format(Omega))

  omega_hat4=np.array(omega_hat4,dtype=int)
  merge_omega=np.concatenate((omega_hat4,Omega))
  tem_array=np.zeros(shape=(N,LCMofArray(merge_omega),N,tau_max_pcmci+1))
  shape=tem_array.shape
  print("tem_array.shape={}".format(shape))
  # union_matrix=np.zeros(shape=(N,N,tau_max_pcmci+1))
  omega_hat4=np.array(omega_hat4,dtype=int)
  for k in range(N): #for one specifc variable
    print("The current target variable is {},total #Variable is {}".format(k,N))
    # for i in range(omega_hat4[k]): #i represent the omega_hat for this variable;Omega=2 then i = 1.
    print("The current target variable is {},the Omega_hat is {}".format(k,Omega[k]))
    for j in range(0,omega_hat4[k]): #different parents set; if Omega=2,then i=1, then j=0,1;
      print("The current target variable is {},the set index is {}".format(k,j))
      dataframe = pp.DataFrame(data_run, mask= mask_all[omega_hat4[k]-1][j][N:T,:],
                                datatime = {0:np.arange(len(data_run))}, 
                                var_names=var_names)
      cmi_symb = CMIsymb(significance='shuffle_test',mask_type='y')
      pcmci_cmi_symb = PCMCI(
      dataframe=dataframe, 
      cond_ind_test=cmi_symb,
      verbosity=1)
      pcmci_cmi_symb.verbosity = 1
      results = pcmci_cmi_symb.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci, parents=superset_dict_est, alpha_level=0.1)
      #p matrix = [N*N*(tau_max+1)]
      tem_array[k][j]=results['p_matrix'][:,k,:]<0.01
      tem_array[k][j]=np.asmatrix(tem_array[k][j])
  et = time.time()
  # get the execution time
  elapsed_time = et - st
  return tem_array, omega_hat4, superset_bool, elapsed_time,results_variable
# print('Execution time:', elapsed_time, 'seconds')
