# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:44:23 2022

@author: OMELCHENKO_V
"""

 
import pickle
import numpy as np
import gurobipy as gp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
timeStart = timer()
from gradual_increase_lib import *

connection_params = {
# For Compute Server you need at least this
   "ComputeServer":  "https://gurobi.dev.optimisation.alpiq.services:443",
   "Username": "OMELCHENKO_V",
   "ServerPassword": "s5XrG3pYqgptaq89jZ"
    }
env = gp.Env(params=connection_params)
#env = []

a_file = open("Assets150.pkl", "rb")
assets_biogas = pickle.load(a_file)
a_file = open("INPUTS_Price_RHS.pkl", "rb")
opti_input = pickle.load(a_file)

Frc    = opti_input['Price']
RHS_cpl  = opti_input['RHS'] #apart from 'RHS' you can also choose 'RHS1', 'RHS2', 'RHS3', 'RHS4'
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
N_initial_portfolio    = 42
# Initial sub-pool with 9 assets:
WS, M0   =      GI_Scratch(N_initial_portfolio, 50, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1 =  GI_WS(N_entire_portfolio, 150, WS,    N_initial_portfolio, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# Calculation without GI:
WS2, M2  =       GI_Scratch(N_entire_portfolio, 200, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

#M3, ObjV, xv1, SM1 =  GI_WS1(N_entire_portfolio, 100, 400,    N_initial_portfolio, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

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
Visualize(Smp, Frc, 0, T, 'Cumulative Power of Asset '+str(AssetN), 'Price', 'Schedule Against Prices for Asset '+str(AssetN), 0)
# The visualization of the total sum against prices. 
Visualize(SM1[-1].x, Frc, 0, T, 'Total Cumulative Power', 'Price', 'Schedule Against Prices', 0)

plt.plot(SM1[-1].x)
plt.plot(RHS_cpl)
plt.show()

# How much time it took
timeEnd = timer()
print('The whole calculation took '+STR(timeEnd-timeStart)+' seconds.')

# ObjVAL, M4  =       GI_ScratchDirectSum(N_entire_portfolio, 150, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# WS2,    M2  =       GI_Scratch(N_entire_portfolio, 150, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

#Best objective 2.984077340076e+08, best bound 2.985095593818e+08, gap 0.0341%


n1 = 25; n2 = 65; n3 = 150

M01, vxA1, SM1   =      GI_ScratchM(n1, 30,  RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

M01.presolve()
M01.optimize()
WS = []
for j in range(n1):
    WS.append(vxA1[j].x)
M02, vxA2, SM2   =      GI_ScratchM(n2, 90,  RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
M02.presolve()
M02A, xv2A = AddWarmStart(M02, assets_biogas, vxA2, WS, n1, n2)
M02A.optimize()
WS = []
for j in range(n2):
    WS.append(xv2A[j].x)

M03, vxA3, SM2   =      GI_ScratchM(n3, 80, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

M03.presolve()
M03A, xv3A = AddWarmStart(M03, assets_biogas, vxA3, WS, n2, n3)
M03A.optimize()




print('MIPgap of BF   = '+STR(M2.MIPgap*100)+'%')
print('MIPgap of GI0  = '+STR(M1.MIPgap*100)+'%.  Two Sub-Pools:   1-42, 1-150.')
print('MIPgap of GI1  = '+STR(M03A.MIPgap*100)+'%.  Three Sub-Pools: 1-37, 1-70, 1-150.')
MIPGAP = abs(max(M1.ObjVal,M2.ObjVal, M03A.ObjVal)-min(M1.ObjBound,M2.ObjBound, M03A.ObjBound))/max(M1.ObjVal,M2.ObjVal, M03A.ObjBound)
print('SynergicGap    = '+STR(MIPGAP*100)+'%.  Min available ub & Max avalable lb.')
# Calculation time
print('GI took '+str(M0.RunTime + M1.RunTime))
print('BF took '+str(M2.RunTime) )
timeEnd = timer()
print('The whole calculation took '+STR(timeEnd-timeStart)+' seconds.')

print([M03.ObjVal, M1.ObjVal])








#M04A = M03A


print(' MIPgap of BF   = '+STR(M2.MIPgap*100)+'%')
print(' MIPgap of GI0  = '+STR(M1.MIPgap*100)+'%.  Two Sub-Pools:   1-42, 1-150.')
print(' MIPgap of GI1  = '+STR(M03A.MIPgap*100)+'%.  Three Sub-Pools: 1-'+str(n1)+', 1-'+str(n2)+', 1-150.')
print(' MIPgap of GI2  = '+STR(M04A.MIPgap*100)+'%.  Three Sub-Pools: 1-35, 1-75, 1-150.')

MIPGAP = abs(max(M1.ObjVal,M2.ObjVal, M03A.ObjVal, M04A.ObjVal)-min(M1.ObjBound,M2.ObjBound, M03A.ObjBound, M04A.ObjBound))/max(M1.ObjVal,M2.ObjVal, M03A.ObjBound, M04A.ObjBound)
print(' Synergy_Gap    = '+STR(MIPGAP*100)+'%.  Min available ub & Max avalable lb.')
# Calculation time
print('GI took '+str(M0.RunTime + M1.RunTime))
print('BF took '+str(M2.RunTime) )
timeEnd = timer()
print('The whole calculation took '+STR(timeEnd-timeStart)+' seconds.')









n1 = 25; n2 = 65; n3 = 100; n4 = 150

M01, vxA1, SM1   =      GI_ScratchM(n1, 30,  RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

M01.presolve()
M01.optimize()
WS = []
for j in range(n1):
    WS.append(vxA1[j].x)
M02, vxA2, SM2   =      GI_ScratchM(n2, 90,  RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
M02.presolve()
M02A, xv2A = AddWarmStart(M02, assets_biogas, vxA2, WS, n1, n2)
M02A.optimize()
WS = []
for j in range(n2):
    WS.append(xv2A[j].x)

M03, vxA3, SM2   =      GI_ScratchM(n3, 75, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

M03.presolve()
M03A, xv3A = AddWarmStart(M03, assets_biogas, vxA3, WS, n2, n3)
M03A.optimize()


M04, vxA4, SM2   =      GI_ScratchM(n4, 30, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

M04.presolve()
M04A, xv4A = AddWarmStart(M04, assets_biogas, vxA4, WS, n3, n4)
M04A.optimize()
