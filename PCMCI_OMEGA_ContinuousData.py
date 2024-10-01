# FILENAME:  PCMCI_OMEGA_conti.py
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
import tigramite
import scipy
import math
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import independence_tests
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction
import random
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy import nan
from math import isnan
from copy import deepcopy
import os
# os. getcwd() 
os.chdir('/home/gao565/PCMCI_OMEGA_github')
# print("work?")
################################################Data Simulations#####################################
def data_generate(T,N,Omega_bound,tau_max,tau_bound,complexity):
  Omega = [0]*N
  for i in range(N):
      Omega[i]=random.randint(1,Omega_bound)
  if max(Omega)<Omega_bound:
    selected_one = random.randint(1,N)
    Omega[i]=Omega_bound
  #np.savetxt("Omega.csv", Omega, delimiter=",")
  true_edge_array=np.zeros(shape=(N,int(np.max(Omega)),N,tau_bound+1))
  for i in range(N):#the target varible
    for j in range(Omega[i]):#the different parents set index
      for k in range(N):#the parents variable
          for s in range(1,tau_max+1):
            if (k==i and s==1):
              true_edge_array[i][j][k][s]=1
            elif s!=tau_max:
              true_edge_array[i][j][k][s]=np.random.binomial(1, complexity, size=None)
            elif s==tau_max:
              h= random.randint(0,N)
              if k==h:
                true_edge_array[i][j][k][s]=1
              elif k!=h:
                true_edge_array[i][j][k][s]==np.random.binomial(1, complexity, size=None)
  true_edge_coef=np.zeros(shape=(N,int(np.max(Omega)),N,tau_bound+1))
  for i in range(N):#the target varible
    for j in range(Omega[i]):#the different parents set index
      for k in range(N):#the parents variable
          for s in range(tau_bound+1):
            if true_edge_array[i][j][k][s]==1:
              true_edge_coef[i][j][k][s]=random.uniform(-1, 1)
  true_edge_coef = np.array(true_edge_coef,dtype=object) 
  for i in range(N):#the target varible
    for is_1 in range(Omega[i]):
      for is_2 in range(Omega[i]):
        if sum(true_edge_array[i,is_1]!=true_edge_array[i,is_2])==0: # Two parents set are exactly same
          true_edge_coef[i,is_1]==true_edge_coef[i,is_2] # then the corresponding coefficient should be also exactly same
  mu = [0]*N
  for i in range(N):
      mu[i] = random.uniform(-1,1) 
  #np.savetxt("mu.csv",mu, delimiter=",")

  sigma = [0]*N
  for i in range(N):
      sigma[i] = random.uniform(0,1)
  #np.savetxt("sigma.csv",sigma, delimiter=",")

  # sigma0=random.uniform(-1,1)
  # mu0=random.uniform(-1,1)
  data = [ [0]*N for i in range(T)]
  data=np.array(data,dtype=float)
  for i in range(N):
      data[:,i] = np.transpose(np.random.randn(T, 1))
      #Or data[:,i] = np.transpose(np.random.randn(T, 1)*sigma0+ mu0)
  #print(data)
  U = range(T)
  for t in range(int(np.max(tau_bound)),T):
    for i in range(0,N):
      for j in range(0,Omega[i]):
        if U[t]%Omega[i]==j:
          for s in range(1,tau_bound+1):
            data[t,i] +=np.matmul(true_edge_coef[i,j,:,s],data[t-s,:])
  return data,Omega,true_edge_array,true_edge_coef,sigma,mu

def data_generate_update(T,N,Omega_bound,tau_max,tau_bound,complexity):
  while True:
    try:
      data_result=data_generate(T,N,Omega_bound,tau_max,tau_bound,complexity)
      data=data_result[0]
      Omega=data_result[1]
      true_edge_array=data_result[2]
      true_edge_coef=data_result[3]
      sigma=data_result[4]
      mu=data_result[5]
    except:
      print("Something went wrong")
      continue
    else:
      if abs(max(data[T-1,:]))>10**20 or sum(np.isnan(data))>1:
        print("Data set is inappropriate")
        continue
      else:
        print("Data set is appropriate")
        break
  return data,Omega,true_edge_array,true_edge_coef,sigma,mu
#@title 
def visualization(N,data):
  fig, axs = plt.subplots(N)
  fig.suptitle('Time series plot for variables')
  for i in range(N):
    axs[i].plot(data[:,i])
    plt.setp(axs[i], yticks=[],ylabel='X{}'.format(i))
  # plotting the line 2 points 
    plt.xlabel('Time index')

################################################Algorithm for Continuous Data#####################################
#Note:  1. function "algorithm_v2_mci_" is the main function for PCMCI_Omega with "turning point" strategy. 
#          The function inputs are "data", "\tau_{ub}" and "\omega_{ub}". "data" has shape [T,N] where T is the time length and N denotes the N-var time series.
#          The function output "tem_array" is the estimated edge array with shape [N,\Omega,N,tau_max_pcmci+1] where the explanation of \Omega can be found in paper.
#          With the same logic in tigramite package, if the target variable has index j, then the incoming edge array is [N,\Omega,j,tau_max_pcmci+1].
#          The function output "omega_hat4" is the estimated periodicity array for each time series.
#          The function output "superset_bool" is the estimated super set of parents set obtained from PCMCI, denoted by "\hat{SPA}" in paper.
#       2. function "algorithm_v2_mci_withoutturningpoint" is the main function of PCMCI_Omega without "turning point" strategy.
#       3. function "PCMCI_omegaV2_results" is the evaluation function for PCMCI_Omega, hence data information is needed.
#       4. function "PCMCI_result" is the evaluation function for PCMCI, hence data information is needed.
#       5. This code is based on tigramite package version v5.1.0. Updates for a newer version of the tigramite package will be available at a later time.
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

def est_summary_casaul_graph(ar,icml_of_true_and_est,omega,N):
  new_ar= np.zeros(shape=(N,icml_of_true_and_est,N,tau_max_pcmci+1)) #if icml_of_ture_and_est=2, omega=2 new_ar=N,2,N,tau+1
  for i in range(N):
    omega_single=omega[i] #omega_single=2
    new_ar[i][0:omega_single]=ar[i][0:omega_single] #0:2=0:2
    replicate_num=int(icml_of_true_and_est/omega_single) #=1
    if replicate_num!=1:
      for j in range(replicate_num-1): #j=0
        new_ar[i][omega_single+j*omega_single:omega_single+(j+1)*omega_single]=ar[i][0:omega_single] #2:4=0:2
      #int(icml_of_true_and_est/omega_hat_single)
  return new_ar

#tau_max_pcmci = "\tau_{ub}" in the paper
#search_omega_bound = "\omega_{ub}" in the paper
def algorithm_v2_mci_(data,T,N,tau_max_pcmci,search_omega_bound):
  st = time.time()
  mask_all=np.empty(shape=(search_omega_bound,search_omega_bound,T,N))
  mask_all=mask_all+3 # The "+3" is a precaution in case there's an issue with generation, which would otherwise cause an error.

  for i in range(1, search_omega_bound+1):
    for j in range(0,i):
      a = [ [1]*N for i in range(T)]
      a =np.array(a,dtype=float)
      U = range(T)
      for t in range(N,T):
        if U[t]%i==j:
          a[t,:]=0
      mask_all[i-1][j]=a

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
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))}, 
                            var_names=var_names)
  parcorr = ParCorr(significance='analytic')
  pcmci = PCMCI(
      dataframe=dataframe, 
      cond_ind_test=parcorr,
      verbosity=0)
  pcmci.verbosity = 0
  results = pcmci.run_pcmci(tau_min=1,tau_max=tau_max_pcmci, pc_alpha=None, alpha_level=0.01)

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

  #Superset_dict_est will be used in MCI tests and then select the Omega_hat.
  for i in range(1,search_omega_bound+1):#search omega=1,2
    print("The current omega is {},total #Omega is {}".format(i,search_omega_bound))
    for j in range(0,i):
      print("The current sub_sample is {},total #sub_sample under omega= {} is {}".format(j,i,i))
      dataframe = pp.DataFrame(data_run, mask= mask_all[i-1][j][N:T,:],
                              datatime = {0:np.arange(len(data_run))}, 
                              var_names=var_names)
      parcorr = ParCorr(significance='analytic',mask_type='y')
      pcmci = PCMCI(
          dataframe=dataframe, 
          cond_ind_test=parcorr,
          verbosity=0)
      pcmci.verbosity = 0
      results = pcmci.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci, parents=superset_dict_est , alpha_level=0.01)
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
      parcorr = ParCorr(significance='analytic',mask_type='y')
      pcmci = PCMCI(
          dataframe=dataframe, 
          cond_ind_test=parcorr,
          verbosity=0)
      pcmci.verbosity = 0
      #superset_dict_est
      results = pcmci.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
      #results = pcmci.run_mci(tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
      #p matrix = [N*N*(tau_max+1)]
      tem_array[k][j]=results['p_matrix'][:,k,:]<0.01
      tem_array[k][j]=np.asmatrix(tem_array[k][j])

  et = time.time()
  # get the execution time
  elapsed_time = et - st
  return tem_array, omega_hat4, superset_bool, elapsed_time


def algorithm_v2_mci_withoutturningpoint(data,T,N,tau_max_pcmci,search_omega_bound):
  st = time.time()
  mask_all=np.empty(shape=(search_omega_bound,search_omega_bound,T,N))
  mask_all=mask_all+3
  for i in range(1, search_omega_bound+1):
    for j in range(0,i):
      a = [ [1]*N for i in range(T)]
      a =np.array(a,dtype=float)
      U = range(T)
      for t in range(N,T):
        if U[t]%i==j:
          a[t,:]=0
      mask_all[i-1][j]=a

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
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))}, 
                            var_names=var_names)
  parcorr = ParCorr(significance='analytic')
  pcmci = PCMCI(
      dataframe=dataframe, 
      cond_ind_test=parcorr,
      verbosity=0)
  pcmci.verbosity = 0
  results = pcmci.run_pcmci(tau_min=1,tau_max=tau_max_pcmci, pc_alpha=None, alpha_level=0.01)

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
      parcorr = ParCorr(significance='analytic',mask_type='y')
      pcmci = PCMCI(
          dataframe=dataframe, 
          cond_ind_test=parcorr,
          verbosity=0)
      pcmci.verbosity = 0
      results = pcmci.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci, parents=superset_dict_est , alpha_level=0.01)
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
  omega_hat4=omega_hat2
  print("Omega hat is {}".format(omega_hat4))
  print("Omega is {}".format(Omega))
  omega_hat4=np.array(omega_hat4,dtype=int)
  merge_omega=np.concatenate((omega_hat4,Omega))
  tem_array=np.zeros(shape=(N,LCMofArray(merge_omega),N,tau_max_pcmci+1))
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
      parcorr = ParCorr(significance='analytic',mask_type='y')
      pcmci = PCMCI(
          dataframe=dataframe, 
          cond_ind_test=parcorr,
          verbosity=0)
      pcmci.verbosity = 0
      #superset_dict_est
      results = pcmci.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
      #results = pcmci.run_mci(tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
      #p matrix = [N*N*(tau_max+1)]
      tem_array[k][j]=results['p_matrix'][:,k,:]<0.01
      tem_array[k][j]=np.asmatrix(tem_array[k][j])
  et = time.time()
  # get the execution time
  elapsed_time = et - st
  return tem_array, omega_hat4, superset_bool, elapsed_time
# print('Execution time:', elapsed_time, 'seconds')

# PCMCI_omegaV2 results （summary_graph, correct)
def PCMCI_omegaV2_results(omega_hat4,tem_array,Omega,true_edge_array,tau_max_pcmci):
  print(omega_hat4)
  print(Omega)
  merge_omega=np.concatenate((omega_hat4,Omega))
  lcm=LCMofArray(merge_omega)
  summary_matrix=est_summary_casaul_graph(true_edge_array,lcm,Omega,N)

  tem_array1=deepcopy(tem_array)
  est_summary_matrix=est_summary_casaul_graph(tem_array1,lcm,omega_hat4,N)
  print(est_summary_matrix.shape)
  print(summary_matrix.shape)
  print(sum(est_summary_matrix))
  print(sum(summary_matrix))
  metric_matrix=est_summary_matrix-summary_matrix
  False_Negative=sum(metric_matrix == -1) #False Negative
  False_Positive=sum(metric_matrix == 1)  #False Positive
  sum_true=sum(summary_matrix) # Total positive-False Negative=True Positive
  True_Positive = sum_true-False_Negative
  precision=True_Positive/(True_Positive+False_Positive)
  recall=True_Positive/(True_Positive+False_Negative)
  F1_score=True_Positive/(True_Positive+1/2*(False_Positive+False_Negative))
  accurate_rate=np.sum(Omega==omega_hat4)/N
  return precision,recall,F1_score, accurate_rate


# PCMCI results （summary_graph, correct)
def PCMCI_result(superset_bool,Omega,true_edge_array,tau_max_pcmci):
  PCMCI_result = np.zeros(shape=(N,N,tau_max_pcmci+1))
  tem_array2=deepcopy(superset_bool)
  for i in range(N):
    PCMCI_result[i]=tem_array2[:,i,:]
  PCMCI_result=np.expand_dims(PCMCI_result,axis=1)
  PCMCI_result.shape

  omega_PCMCI=np.array(np.zeros(shape=(N))+1,dtype=int)
  merge_omega_2=np.concatenate((omega_PCMCI,Omega))
  lcm_2=LCMofArray(merge_omega_2)


  est_summary_matrix=est_summary_casaul_graph(PCMCI_result,lcm_2,omega_PCMCI,N)
  summary_matrix=est_summary_casaul_graph(true_edge_array,lcm_2,Omega,N)

  print(est_summary_matrix.shape)
  print(summary_matrix.shape)
  print(sum(est_summary_matrix))
  print(sum(summary_matrix))
  metric_matrix=est_summary_matrix-summary_matrix
  False_Negative=sum(metric_matrix == -1) #False Negative
  False_Positive=sum(metric_matrix == 1)  #False Positive
  sum_true=sum(summary_matrix) # Total positive-False Negative=True Positive
  True_Positive = sum_true-False_Negative
  precision=True_Positive/(True_Positive+False_Positive)
  recall=True_Positive/(True_Positive+False_Negative)
  F1_score=True_Positive/(True_Positive+1/2*(False_Positive+False_Negative))
  # print("est_summary_matrix={}".format(est_summary_matrix))
  # print("sum_true={}".format(sum_true))
  # print("False_Negative={}".format(False_Negative))
  # print("False_Positive={}".format(False_Positive))
  # print("True_Positive={}".format(True_Positive))
  return precision,recall,F1_score
# print(False_Negative)



##################### Setting ######################
tau_max=5
search_omega_bound=10
# search_tau_bound=20
tau_bound = tau_max # assuming we know the true tau_max
tau_max_pcmci=tau_max # assuming we know the true tau_max
N=5
complexity=0.2/N
T=1000
Omega_bound=5 

#################### Run Experiment ################
while True:
  try:
      data_result=data_generate_update(T=T,N=N,Omega_bound=Omega_bound,tau_max=tau_max,tau_bound=tau_bound,complexity=complexity)
      data=deepcopy(data_result[0])
      Omega=data_result[1]
      true_edge_array=data_result[2]
      true_edge_coef=data_result[3]
      sigma=data_result[4]
      mu=data_result[5]
      algorithm_v2_mci_results=algorithm_v2_mci_(data,T,N,tau_max_pcmci,search_omega_bound)
      # algorithm_v2_mci_results=algorithm_v2_mci_withoutturningpoint(data,T,N,tau_max_pcmci,search_omega_bound)
      print("pcmciomega done")
  except:
      print("Algorithm not works for this dataset")
      continue
  else:
      break
tem_array=algorithm_v2_mci_results[0]
omega_hat4=algorithm_v2_mci_results[1]
superset_bool=algorithm_v2_mci_results[2]
times=np.array(algorithm_v2_mci_results[3],dtype=object)

time_pcmci_omega=times
PCMCI_omegaV2=PCMCI_omegaV2_results(omega_hat4=omega_hat4,tem_array=tem_array,Omega=Omega,true_edge_array=true_edge_array,tau_max_pcmci=tau_max_pcmci)
PCMCI_results=PCMCI_result(superset_bool=superset_bool,Omega=Omega,true_edge_array=true_edge_array,tau_max_pcmci=tau_max_pcmci)
Omega_max=np.max(Omega)
sum_true_edge=sum(true_edge_array)

precision_pcmci_omega=PCMCI_omegaV2[0]
recall_pcmci_omega=PCMCI_omegaV2[1]
F1_score_pcmci_omega=PCMCI_omegaV2[2]
accurate_rate_pcmci_omega=PCMCI_omegaV2[3]

precision_pcmci=PCMCI_results[0]
recall_pcmci=PCMCI_results[1]
F1_score_pcmci=PCMCI_results[2]


Omega_str=np.array(Omega,dtype=str)
OMEGA=';'.join(Omega_str)
omega_hat4_str=np.array(omega_hat4,dtype=str)
omega_hat=';'.join(omega_hat4_str)
T_index=T
N_index=N
tau_index=tau_max

results_all=np.array([T_index,N_index,tau_index,OMEGA,omega_hat,sum_true_edge,Omega_max,time_pcmci_omega,precision_pcmci_omega,recall_pcmci_omega,F1_score_pcmci_omega,accurate_rate_pcmci_omega,precision_pcmci,recall_pcmci,F1_score_pcmci])
results_frame=pd.DataFrame.transpose(pd.DataFrame(results_all))
results_frame.columns =['T','N','tau_max','OMEGA','omega_hat','sum_true_edge','Omega_max','time_pcmci_omega', 'precision_pcmci_omega', 'recall_pcmci_omega', 'F1_score_pcmci_omega','accurate_rate_pcmci_omega','precision_pcmci','recall_pcmci','F1_score_pcmci']
head_str='T,N,tau_max,OMEGA,omega_hat,sum_true_edge,Omega_max,time_pcmci_omega,precision_pcmci_omega, recall_pcmci_omega, F1_score_pcmci_omega,accurate_rate_pcmci_omega,precision_pcmci,recall_pcmci,F1_score_pcmci'
# '%s %s %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e'
np.savetxt('conti_results'+'_omega'+str(Omega_bound)+'T'+str(T)+'N'+str(N)+'tau_max'+str(tau_max)+'.csv',results_frame, delimiter=',',header=head_str,fmt='%s')
