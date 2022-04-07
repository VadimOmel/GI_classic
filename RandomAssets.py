# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:44:27 2022

@author: OMELCHENKO_V
"""


import copy
import pickle
import numpy as np
import gurobipy as gp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
timeStart = timer()
#from q_Opti3 import *
from gurobipy import GRB, Model, Env
from gradual_increase_lib import *
connection_params = {
   "ComputeServer":  "https://gurobi.dev.optimisation.alpiq.services:443",
   "Username": "OMELCHENKO_V",
   "ServerPassword": "s5XrG3pYqgptaq89jZ"
    }
env = gp.Env(params=connection_params)
#env = []
timeStart = timer()

from statsmodels.tsa.arima_process import ArmaProcess
ar1 = np.array([0.5, 0.4])
ma1 = np.array([1, 0.6])
T = 80
simulated_ARMA_data = ArmaProcess(ar1, ma1).generate_sample(nsample=T)
#plt.plot(simulated_ARMA_data - np.min(simulated_ARMA_data))


Frc = simulated_ARMA_data - np.min(simulated_ARMA_data)
Frc = Frc + np.sin(np.cumsum(np.ones(T))/5)*max(Frc)/2
Frc = Frc - min(Frc)-5
plt.plot(Frc)
plt.show()
def subDict(flp, a1, a2):
    return {k: flp[k] for k in (a1, a2) if k in flp}

def Index_Of_Biogas_PP(Nn):
    return 'Biogas'+str(Nn)
def RandomNumber_Of_Turbines(m1,d):
    A=np.random.randint(m1, m1+d, 1)
    return A[0]

def Random_Int_Number(m1,d):
    A=np.random.randint(m1, m1+d, 1)
    return A[0]
    
def Random_Cont_Number(m1,d):
    A=np.random.uniform(m1, m1+d, 1)
    return A[0]

def GenerateTurbines(N_turbs):
    d3 = {}
    for j in range(N_turbs):
        Pmin = Random_Int_Number(200,280)
        d4 = {'Pmin': Pmin,
              'Pmax': 2*Pmin,
              'StartCost': Random_Int_Number(2,11),
              'StopCost':  Random_Int_Number(2,11),
              'TminOn':    Random_Int_Number(2,6),
              'TminOff':   Random_Int_Number(2,7)}
        d3.update({'Turbine'+str(j+1) : d4})
    return d3

def Build_Asset_on_Top(assets, Number_Additional_Assets):
    n1 = len(assets)
    for n in range(Number_Additional_Assets):
        N_turbs = Random_Int_Number(2,7)         
        assets.update({Index_Of_Biogas_PP(n+n1+1): 
                {
                    'Name': 'Asset'+str(n+n1+1),
                    'TechnicalFeature':
                    {
                    'Capacity': 2*Random_Cont_Number(20000,50000),    # kW
                    'GasInflow': 0.5*Random_Cont_Number(400,500),
                    'N_Turbines': N_turbs,
                    'Turbines': GenerateTurbines(N_turbs)
        
                    }
                }
            })    
    return assets



def Capacity_Function(assets):
    CP = 0
    for asset_1 in assets.keys():
        CP = CP + assets[asset_1]['TechnicalFeature']['Capacity']
    return CP

def GasInflow_Function(assets):
    CP = 0
    for asset_1 in assets.keys():
        CP = CP + assets[asset_1]['TechnicalFeature']['GasInflow']
    return CP

assets = Build_Asset_on_Top({}, 150)
T = len(Frc)


RHS = np.zeros(T)
for j in range(200):
    m1 = Random_Int_Number(2,6)
    m2 = m1 + Random_Int_Number(2,4)
    if np.mod(j,2)==0:
        m3 = m2
    else:
        m3 = 1/m2
    RHS = RHS + m3*np.sin(np.cumsum(np.ones(T))/m1)
RHS = RHS + np.sin(np.cumsum(np.ones(T)))
RHS = positiv(RHS)
CAP = GasInflow_Function(assets)*0.1
RHS = RHS/max(RHS)*CAP 
plt.plot(RHS)
plt.show()

RHS_cpl  = np.flip(RHS)

L01  = 0
PowerPl = 'Biogas1'
variable_input_biogas  = [Frc, RHS_cpl, L01, np.array(CUMSUMV(np.zeros(10), T)), np.array(CUMSUMV(np.zeros(10), T)) ]
variable_input = variable_input_biogas
assets_biogas = assets


assets0 = assets_biogas['Biogas1']
RunOpti = 1
M = []
M1, Smp1, p1, u1, v1, w1, Storage1, Spill1 =                                                 ModelSinglePP_Biogas(    variable_input, assets0, RunOpti, M, env)
M, Ainq, binq, Aeq, beq, OBJf, COUPLINGMatr, Box, u,v,w,p,Smp,Storage,Spill, INTGR, CTIME0 = ModelSinglePP_BiogasMTRX(variable_input, assets0, RunOpti, M, env)
KEYS = list(assets_biogas.keys())
XV = []
RUN_TIMES = []
for asset_1 in KEYS:
    assets0 = assets_biogas[asset_1]
    M, Ainq, binq, Aeq, beq, OBJf, COUPLINGMatr, Box, u,v,w,p,Smp,Storage,Spill, INTGR, CTIME0 = ModelSinglePP_BiogasMTRX(variable_input, assets0, RunOpti, M, env)
    XV.append(M.x)
    RUN_TIMES.append(M.RunTime)
    print(asset_1)

"""Preparation of matrices and vectors for the pool of assets"""

AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx = PrepareMatrixInputs(assets_biogas, variable_input, env)
print('The start of Gradual Increase')
i = 0
for asset_1 in assets_biogas.keys():
    assets_biogas[asset_1].update({'SCH': np.array(XV[i])})
    i = i + 1
    
N_entire_portfolio     = len(assets_biogas)

Time3    = 200
Time1    = 50
Time2    = Time3-Time1
# Calculation without GI:
WS2, M2  =       GI_Scratch(N_entire_portfolio, Time3, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)


# GI
N_initial_portfolio    = 42
# Initial sub-pool with 9 assets:
WS, M0   =      GI_Scratch(N_initial_portfolio, Time1, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1 =  GI_WS(N_entire_portfolio, Time2, WS,    N_initial_portfolio, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

print('MIPgap of GI = '+STR(M1.MIPgap*100)+'%')
print('MIPgap of BF = '+STR(M2.MIPgap*100)+'%')
MIPGAP = abs(max(M1.ObjVal,M2.ObjVal)-min(M1.ObjBound,M2.ObjBound))/max(M1.ObjVal,M2.ObjVal)
print('SynergicGap  = '+STR(MIPGAP*100)+'%')
# Calculation time
print('GI took '+str(M0.RunTime + M1.RunTime))
print('BF took '+str(M2.RunTime) )

plt.plot(SM1[-1].x)
plt.plot(RHS_cpl)
plt.ylim([0, 60000])

timeEnd = timer()

print('It took '+str(timeEnd - timeStart)+' seconds')








# WS, M0   =      GI_Scratch(15, 20, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# # WS is the output of GI_Scratch. 
# # It is used as the warm start:
# M1, ObjV, xv1, SM1, WS =  GI_WS2(20, 20, WS,     15, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

# M1, ObjV, xv1, SM1, WS =  GI_WS2(30, 20, WS,     20, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

# M1, ObjV, xv1, SM1, WS =  GI_WS2(40, 135, WS,    30, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

# M1, ObjV, xv1, SM1, WS =  GI_WS2(60, 135, WS,    40, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)















# WS, M0   =      GI_Scratch(15, 20, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# # WS is the output of GI_Scratch. 
# # It is used as the warm start:
# M1, ObjV, xv1, SM1, WS =  GI_WS2(30, 30, WS,     15, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

# M1, ObjV, xv1, SM1, WS =  GI_WS2(60, 245, WS,     30, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

# N_initial_portfolio    = 64
# # Initial sub-pool with 9 assets:
# M0, xv0, SM0   =      GI_ScratchM(N_initial_portfolio, 35, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# # WS is the output of GI_Scratch. 
# # It is used as the warm start:

# # GI
# # Initial sub-pool with 9 assets:
# WS, M00   =      GI_Scratch(N_initial_portfolio, 135, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# # WS is the output of GI_Scratch. 
# # It is used as the warm start:    


# M1, xv, SM     =      GI_ScratchM(N_entire_portfolio,  135, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

# M1.presolve()
# for j in range(N_entire_portfolio):
#     ASSET = KEYS[j]
#     strts  = assets_biogas[ASSET]['SCH']
    
#     if j<=N_initial_portfolio - 1:
#          strts2 = WS[j]
#          xv[j].start = np.array(strts2)
#     else:
#          xv[j].start = np.array(strts)

# M1.optimize()


# M1A, xvA, SMA     =      GI_ScratchM(N_entire_portfolio,  135, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)



# def AddWarmStart(M, assets_biogas, xv, WS, N_initial_portfolio, N_entire_portfolio):
#     KEYS = list(assets_biogas.keys())
#     for j in range(N_entire_portfolio):
#         ASSET = KEYS[j]
#         strts  = assets_biogas[ASSET]['SCH']
        
#         if j<=N_initial_portfolio - 1:
#              strts2 = WS[j]
#              xv[j].start = np.array(strts2)
#         else:
#              xv[j].start = np.array(strts)  
#     return M, xv

# n1 = 37; n2 = 50; n3 = 80; n4 = 150

# M01, vxA1, SM1   =      GI_ScratchM(n1, 15, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# M02, vxA2, SM2   =      GI_ScratchM(n2, 20, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# M03, vxA3, SM2   =      GI_ScratchM(n3, 30, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# M04, vxA4, SM3   =      GI_ScratchM(n4, 135, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

# M01.presolve()
# M01.optimize()
# WS = []
# for j in range(n1):
#     WS.append(vxA1[j].x)

# M02.presolve()
# M02A, xv2A = AddWarmStart(M02, assets_biogas, vxA2, WS, n1, n2)
# M02A.optimize()
# WS = []
# for j in range(n2):
#     WS.append(xv2A[j].x)


# M03.presolve()
# M03A, xv3A = AddWarmStart(M03, assets_biogas, vxA3, WS, n2, n3)
# M03A.optimize()
# WS = []
# for j in range(n3):
#     WS.append(xv3A[j].x)
    

# M04.presolve()
# M04A, xv4A = AddWarmStart(M04, assets_biogas, vxA4, WS, n3, n4)
# M04A.optimize()
# # WS = []
# # for j in range(n3):
# #     WS.append(xv3A[j].x)
# M04, vxA4, SM3   =      GI_ScratchM(n4, 70+50+40+30, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)






# #RHS_cpl1 = RHS_cpl





# n1 = 37; n2 = 70; n3 = 150

# M01, vxA1, SM1   =      GI_ScratchM(n1, 30, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# M02, vxA2, SM2   =      GI_ScratchM(n2, 20, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# M03, vxA3, SM2   =      GI_ScratchM(n3, 150, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)

# M01.presolve()
# M01.optimize()
# WS = []
# for j in range(n1):
#     WS.append(vxA1[j].x)

# M02.presolve()
# M02A, xv2A = AddWarmStart(M02, assets_biogas, vxA2, WS, n1, n2)
# M02A.optimize()
# WS = []
# for j in range(n2):
#     WS.append(xv2A[j].x)


# M03.presolve()
# M03A, xv3A = AddWarmStart(M03, assets_biogas, vxA3, WS, n2, n3)
# M03A.optimize()




# print('MIPgap of GI  = '+STR(M1.MIPgap*100)+'%')
# print('MIPgap of GI1 = '+STR(M03A.MIPgap*100)+'%')
# print('MIPgap of BF  = '+STR(M2.MIPgap*100)+'%')
# MIPGAP = abs(max(M1.ObjVal,M2.ObjVal, M03A.ObjVal)-min(M1.ObjBound,M2.ObjBound, M03A.ObjBound))/max(M1.ObjVal,M2.ObjVal, M03A.ObjBound)
# print('SynergicGap   = '+STR(MIPGAP*100)+'%')
# # Calculation time
# print('GI took '+str(M0.RunTime + M1.RunTime))
# print('BF took '+str(M2.RunTime) )


# PriceAndRHS = {'Price': Frc}
# PriceAndRHS.update({'RHS': RHS_cpl})

# a_file = open("INPUTS_Price_RHS.pkl", "wb")
# pickle.dump(PriceAndRHS, a_file)
# a_file.close()


# a_file = open("Assets150.pkl", "wb")
# pickle.dump(assets_biogas, a_file)
# a_file.close()


