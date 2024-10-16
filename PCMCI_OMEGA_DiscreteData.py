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
import os
# os. getcwd() 
os.chdir('/home/gao565/PCMCI_OMEGA_github')
# print("work?")
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

def discrete_data_generate(T,N,Omega_bound,tau_max,tau_bound,complexity):
# twodbn=gum.fastBN("a0->at;a0->bt;a0->ct;b0->bt;b0->ct;c0->ct;c0->et;d0->dt;d0->et;e0->et;",2)
# gdyn.showTimeSlices(twodbn)
# twodbn=gum.fastBN("a0->at;b0->bt;c0->ct;d0->dt;e0->et",2)
  # Generate a dynamic Bayesian network based on N
  edges = ";".join([f"{chr(97 + i)}0->{chr(97 + i)}t" for i in range(N)])  # e.g., "a0->at;b0->bt;...;e0->et"
  twodbn = gum.fastBN(edges, 2)
  gdyn.showTimeSlices(twodbn)
  #Initialize the time series without any edges
  dbn=gdyn.unroll2TBN(twodbn,T)
  # gdyn.showTimeSlices(dbn,86)
  # new_key=["a","b","c","d","e"]
  # Generate new_key dynamically based on N
  new_key = [chr(97 + i) for i in range(N)] 
  for i in range(int(np.max(tau_bound)),T):
    for j in range(N):
      # dbn.cpt(f"c{i}").fillWith(pot,["ct","c0"]) # ct in pot <- first var of cpt, c0 in pot<-second var in cpt
      lag=i-1
      dbn.eraseArc(f"{new_key[j]}{lag}",f"{new_key[j]}{i}")
  # gdyn.showTimeSlices(dbn,size="84")
  #The largest Omega in experiment is 15;
  tau_max_pcmci=tau_max
  Omega = [0]*N
  for i in range(N):
      Omega[i]=random.randint(1,Omega_bound)
  #np.savetxt("Omega.csv", Omega, delimiter=",")
  print(Omega)
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
  icml_of_true_and_est=LCMofArray(Omega)
  expand_true_edge_array = est_summary_casaul_graph(ar=true_edge_array,icml_of_true_and_est=icml_of_true_and_est, omega=Omega, N=N)

  # new_key=["a","b","c","d","e"]
  var_dic={}
  for s in range(icml_of_true_and_est):
    var_dic[s]={}
  for s in range(icml_of_true_and_est):
    for i in range(N):
      var_dic[s][new_key[i]] = []
  for s in range(icml_of_true_and_est):
    for k in range(N):
      for i in range(N):
        for j in range(tau_bound+1):
          if expand_true_edge_array[k,s,i,j]==1:
            var_dic[s][new_key[k]].append((new_key[i],-j))

  #Initialize the time series without any edges
  dbn=gdyn.unroll2TBN(twodbn,T)
  # gdyn.showTimeSlices(dbn,86)
  # new_key=["a","b","c","d","e"]
  for i in range(int(np.max(tau_bound)),T):
    for j in range(N):
      # dbn.cpt(f"c{i}").fillWith(pot,["ct","c0"]) # ct in pot <- first var of cpt, c0 in pot<-second var in cpt
      lag=i-1
      dbn.eraseArc(f"{new_key[j]}{lag}",f"{new_key[j]}{i}")
  # gdyn.showTimeSlices(dbn,size="84")
  cols=N
  rows=icml_of_true_and_est
  # distributon_ar =[[0]*cols]*rows: A large trap: It's because * is copying the address of the object (list).
  # a = [[0]*3]*3
  # a
  # [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  # a[0][0]=1
# a
# [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
# distributon_ar =[[0,0,0,0,0],[0,0,0,0,0]]
  distributon_ar = [ [0]*N for i in range(icml_of_true_and_est)]
  for k in range(N):
    for s in range(icml_of_true_and_est):
      # print("k={},s={},len={}".format(k,s,len(var_dic[s][new_key[k]])))
      len_ar=pow(2,len(var_dic[s][new_key[k]])+1)
      distributon_ar[s][k]=list(np.random.uniform(0,1,len_ar))
      # print(len(distributon_ar[s][k]))
      # print(distributon_ar[s][k])

  distributon_new_ar=deepcopy(distributon_ar)
  for k in range(N):
    if Omega[k]<icml_of_true_and_est:
      omega_single=Omega[k]
      # print(omega_single)
      for h in range(omega_single):
        distributon_new_ar[h][k]=distributon_ar[h][k]
      replicate_num=int(icml_of_true_and_est/omega_single)
      if replicate_num!=1:
        for j in range(replicate_num-1): #j=0
          start=omega_single+j*omega_single
          end=omega_single+(j+1)*omega_single
          for q in range(start,end):
            p=q-(j+1)*omega_single
            # print("p={},q={}".format(p,q))
            distributon_new_ar[q][k]=distributon_ar[p][k] #2:4=0:2

    for i in range(N):#the target varible
      for is_1 in range(Omega[i]):
        for is_2 in range(Omega[i]):
          if sum(true_edge_array[i,is_1]!=true_edge_array[i,is_2])==0: # Two parents set are exactly same
            distributon_new_ar[is_1][i]==distributon_ar[is_1][i] # then the corresponding coefficient should be also exactly same
  U = range(T)
  for t in range(int(np.max(tau_bound)),T):
    for k in range(N):
      pot=gum.Potential().add(dbn.variableFromName(f"{new_key[k]}{t}"))
      for s in range(icml_of_true_and_est):
        if U[t]%icml_of_true_and_est==s:#0/6=0;1/6=1;....,5/6=5,6/6=0,7/6=1,...,
          for j in range(len(var_dic[s][new_key[k]])):
            lag=t+var_dic[s][new_key[k]][j][1]
            # print(f"{var_dic[s][new_key[k]][j][0]}{lag}"+f"{new_key[k]}{t}")
            dbn.addArc(f"{var_dic[s][new_key[k]][j][0]}{lag}",f"{new_key[k]}{t}")
            pot.add(dbn.variableFromName(f"{var_dic[s][new_key[k]][j][0]}{lag}"))
    # dbn.cpt(f"b{t}").fillWith(pot,[f"b{t}","b0"]) # ct in pot <- first var of cpt, c0 in pot<-second var in cpt
        # len_ar=pow(3,len(var_dic[s][new_key[k]])+1)
        # temp_ar=np.random.uniform(0,1,len_ar)
          # print("t={},k={},s={},j={}".format(t,k,s,j))
          # print(len(distributon_ar[s][k]))
          pot.fillWith(distributon_ar[s][k]).normalizeAsCPT()
          # print(pot)
          dbn.cpt(f"{new_key[k]}{t}").fillWith(pot)
  # generating complete date with pyAgrum
  size=1
  # dbn=gdyn.unroll2TBN(twodbn,T)
  generator=gum.BNDatabaseGenerator(dbn)
  # generator.setTopologicalVarOrder()
  generator.drawSamples(1)
  completeData="discrete_data_tem.csv"
  generator.toCSV(completeData)
  t=pd.read_csv(completeData,header=None)
  t=pd.DataFrame.transpose(t)
  t.columns=["col_1","col_2"]
# Initialize a list to hold the variables
  vars = []

  # Collect data for each variable at intervals of T
  for i in range(N):
      start_index = i * T
      end_index = (i + 1) * T
      var = t['col_2'][start_index:end_index]  # Collect data every T points
      vars.append(var)
      print(f"var{i}.shape: {var.shape}")  # Print the shape of each variable

  # Convert the list to a NumPy array
  data = np.array(vars)
  data=pd.DataFrame.transpose(pd.DataFrame(data))
  data=np.array(data,dtype=int)
  np.savetxt('data.csv',data, delimiter=',',fmt='%s')

  return data, Omega,true_edge_array,var_dic



def algorithm_v2_mci_(data,T,N,tau_max_pcmci,search_omega_bound):
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
      results = pcmci_cmi_symb.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci, parents=superset_dict_est , alpha_level=0.01)
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
      # parcorr = ParCorr(significance='analytic',mask_type='y')
      # pcmci = PCMCI(
      #     dataframe=dataframe, 
      #     cond_ind_test=parcorr,
      #     verbosity=0)
      # pcmci.verbosity = 0
      # results = pcmci.run_mci(selected_links=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
      #results = pcmci.run_mci(tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
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
# print('Execution time:', elapsed_time, 'seconds')
  et = time.time()
  # get the execution time
  elapsed_time = et - st
  return tem_array, omega_hat4, superset_bool, elapsed_time,results_variable

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
##################### Setting ######################
tau_max=3
search_omega_bound=7
# search_tau_bound=20
tau_bound = tau_max # assuming we know the true tau_max
tau_max_pcmci=tau_max # assuming we know the true tau_max
N=3
complexity=0.08
T=12000
Omega_bound=3

#################### Run Experiment ################
while True:
    try:
        data_result=discrete_data_generate(T=T,N=N,Omega_bound=Omega_bound,tau_max=tau_max,tau_bound=tau_bound,complexity=complexity)
        data=deepcopy(data_result[0])
        Omega=data_result[1]
        true_edge_array=data_result[2]
        true_edge_coef=data_result[3]
        print(true_edge_array)
        ## use turning point
        algorithm_v2_mci_results=algorithm_v2_mci_(data,T,N,tau_max_pcmci,search_omega_bound) 
        results_variable=algorithm_v2_mci_results[4]
        ## not use turning point
        #algorithm_v2_mci_resultswithoutturningpoint=algorithm_v2_mci_withoutturningpoint(data,T,N,tau_max_pcmci,search_omega_bound)
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
np.savetxt('discrete_results'+'_omega'+str(Omega_bound)+'T'+str(T)+'N'+str(N)+'tau_max'+str(tau_max)+'.csv',results_frame, delimiter=',',header=head_str,fmt='%s')
