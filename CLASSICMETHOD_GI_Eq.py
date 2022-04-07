# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:22:34 2022

@author: OMELCHENKO_V
"""

import pickle
import numpy as np
import gurobipy as gp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
timeStart = timer()
from gradual_increase_lib_Eq import *

connection_params = {
   "ComputeServer":  "YOUR_SERVER",
   "Username": "YOUR_USERNAME",
   "ServerPassword": "YOUR_PASSWORD"
    }
env = gp.Env(params=connection_params)
#env = []
a_file = open("TestGI.pkl", "rb")
assets_biogas = pickle.load(a_file)
a_file = open("OptiInput.pkl", "rb")
opti_input = pickle.load(a_file)

Frc    = 0*opti_input['Prices']
RHS_cpl  = opti_input['RHS']  #apart from 'RHS' you can also choose 'RHS1', 'RHS2', 'RHS3', 'RHS4'
T    = len(Frc)
L01  = 0
PowerPl = 'Biogas1'
variable_input_biogas  = [Frc, RHS_cpl, L01, np.array(CUMSUMV(np.zeros(10), T)), np.array(CUMSUMV(np.zeros(10), T)) ]
 
""" Demonstration of functions ModelSinglePP_Biogas and ModelSinglePPMTRX.
They have the same input and produce the same objective. The function ModelSinglePPMTRX
also produced matrices for the pool of assets"""
variable_input = variable_input_biogas
RunOpti = 1
assets0 = assets_biogas['Biogas1']
M = []
M1, Smp1, p1, u1, v1, w1, Storage1, Spill1 =                                                 ModelSinglePP_Biogas(    variable_input, assets0, RunOpti, M, env)
M, Ainq, binq, Aeq, beq, OBJf, COUPLINGMatr, Box, u,v,w,p,Smp,Storage,Spill, INTGR, CTIME0 = ModelSinglePP_BiogasMTRX(variable_input, assets0, RunOpti, M, env)
# The comparison of the objective values yielded by the two functions
print('ratio Obj(Mtrx)/Obj(Vrs) = '+STR(100*M.ObjVal/M1.ObjVal)+'%')
"""Demonstration of the total turbines output (Smp) and storage level by two methods"""
# Smp
plt.plot(Smp1.x)
plt.plot(Smp)
plt.title('The total production of all the turbines')
plt.show()

# Storage
plt.plot(Storage1.x)
plt.plot(Storage)
plt.title('The storage levels')
plt.show()

"""Preparation of matrices and vectors for the pool of assets"""
KEYS = list(assets_biogas.keys())
AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx = PrepareMatrixInputs(assets_biogas, variable_input, env)
print('The start of Gradual Increase')
"""The start of Gradual Increase"""
# The initial sub-pool consists of assets 1-9. 
# The second sub-pool contains all assets.
# Comparison of the MIPgap by GI and that of by calculating from scratch
N_entire_portfolio     = len(assets_biogas)
N_initial_portfolio    = 9
# Initial sub-pool with 9 assets:
WS, M0   =      GI_Scratch(N_initial_portfolio, 15, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1 =  GI_WS(N_entire_portfolio, 35, WS,    N_initial_portfolio, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# Calculation without GI:
WS2, M2  =       GI_Scratch(N_entire_portfolio, 50, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

"""The summary"""
# Objectives
print([ObjV, M2.ObjVal])
# MIP gap
print('MIPgap of GI = '+STR(M1.MIPgap*100)+'%')
print('MIPgap of BF = '+STR(M2.MIPgap*100)+'%')
# Calculation time
print('GI took '+str(M0.RunTime + M1.RunTime))
print('BF took '+str(M2.RunTime) )

# The visualization of the output for a randomly chosen asset (Asset number 10)
AssetN = 10
NTurb = len(assets_biogas[KEYS[AssetN]]['TechnicalFeature']['Turbines'])
u,v,w,p,Smp,Storage,Spill = ExtractValues(xv1[AssetN], T, NTurb)
plt.plot(u[0])
plt.plot(v[0])
plt.plot(w[0])
plt.title('Visualization of u,v, and w variables')
plt.show()
plt.plot(Storage)
plt.title('Visualization of the storage levels of Asset '+str(AssetN))
plt.show()
Visualize(Smp, Frc, 0, T, 'Cumulative Power of Asset '+str(AssetN), 'Price', 'Schedule Against Prices', 0)
# The visualization of the total sum against prices. 
Visualize(SM1[-1].x, Frc, 0, T, 'Total Cumulative Power', 'Price', 'Schedule Against Prices', 0)

# How much time it took
timeEnd = timer()
print('The whole calculation took '+STR(timeEnd-timeStart)+' seconds.')

# ObjVAL, M4  =       GI_ScratchDirectSum(N_entire_portfolio, 150, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# WS2,    M2  =       GI_Scratch(N_entire_portfolio, 150, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

