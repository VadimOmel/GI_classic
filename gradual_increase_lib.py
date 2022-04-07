# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 20:10:07 2022

@author: OMELCHENKO_V
l.l,"""


import numpy as np
import gurobipy as gp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
timeStart = timer()
from gurobipy import GRB, Model
from scipy.sparse import coo_matrix, vstack, hstack
import scipy

def STR(val):
    if (val>100):
        return "%0.3f" % val
    else:
        return "%0.4f" % val

def CUMSUMV(vector, N):
    n = len(vector)
    if N<=n:
        N = n
    VcOutp = []
    for j in range(n):
        VcOutp.append(sum(vector[j:]))
    for j in range(n+1,N+1):
        VcOutp.append(0)
    return VcOutp
def Hconcat(M1, M2):
    return np.concatenate((M1           ,  M2),axis=1)

def Vconcat(M1, M2):
    return np.concatenate((M1           ,  M2),axis=0)

def HCONCAT(VCT):
    if len(VCT) == 1:
        return VCT[0]
    else:
        P1 = VCT[0]
        VCT.pop(0)
        for vct in VCT:
            P1 = Hconcat(P1, vct)
        return P1

def VCONCAT(VCT):
    if len(VCT) == 1:
        return VCT[0]
    else:
        P1 = VCT[0]
        VCT.pop(0)
        for vct in VCT:
            P1 = Vconcat(P1, vct)
        return P1


def GI_Scratch(InitN, TL, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    SM = []
    xv = [[]]*InitN
    OBJ = 0
    for j in range(InitN):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
        
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(InitN)+' assets in the pool. Asset '+str(j)+' is added to the model')
            
    M.addConstr(SM[-1] >= Vects)
         
    M.setParam('TimeLimit',TL)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize() 
    
    Xd = []
    for j in range(InitN):
        Xd.append(xv[j].x)
    return Xd, M


def GI_ScratchM(InitN, TL, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    SM = []
    xv = [[]]*InitN
    OBJ = 0
    for j in range(InitN):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
        
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(InitN)+' assets in the pool. Asset '+str(j)+' is added to the model')
    XP = M.addMVar(T, lb=0, ub = 10**9)        
    M.addConstr(SM[-1] + XP >= Vects)
    OBJ = OBJ - 10000*np.ones(T) @ XP     
    M.setParam('TimeLimit',TL)
    M.setObjective(OBJ, GRB.MAXIMIZE)

    return M, xv, SM

def GI_ScratchPen(InitN, TL, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    SM = []
    xv = [[]]*InitN
    OBJ = 0
    for j in range(InitN):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
        
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(InitN)+' assets in the pool. Asset '+str(j)+' is added to the model')
    XP = M.addMVar(tT, lb = 0, ub = 10**8)        
    M.addConstr(SM[-1] + XP >= Vects)
    OBJ = OBJ - 1000*np.ones(T) @ XP
    M.setParam('TimeLimit',TL)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize() 
    
    Xd = []
    for j in range(InitN):
        Xd.append(xv[j].x)
    return Xd, M

def GI_WS(Nlarge, TL2, WS, InitN, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    SM = []
    xv = [[]]*Nlarge
    OBJ = 0
    for j in range(Nlarge):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
        ASSET = KEYS[j]
        strts  = assets_biogas[ASSET]['SCH']
        
        if j<=InitN - 1:
             strts2 = WS[j]
             xv[j].start = np.array(strts2)
        else:
             xv[j].start = np.array(strts)
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(Nlarge)+' assets in the pool. Asset '+str(j)+' is added to the model')
    # variable SM[-1] is the sum of sum_p1 + sum_p2 + ... + sum_pN, where N = number of assets        
    M.addConstr(SM[-1] >= Vects)
         
    M.setParam('TimeLimit',TL2)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize()
    return M, M.ObjVal, xv, SM

def GI_WS2(Nlarge, TL2, WS, InitN, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    SM = []
    xv = [[]]*Nlarge
    OBJ = 0
    for j in range(Nlarge):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
        ASSET = KEYS[j]
        strts  = assets_biogas[ASSET]['SCH']
        
        if j<=InitN - 1:
             strts2 = WS[j]
             xv[j].start = np.array(strts2)
        else:
             xv[j].start = np.array(strts)
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(Nlarge)+' assets in the pool. Asset '+str(j)+' is added to the model')
    # variable SM[-1] is the sum of sum_p1 + sum_p2 + ... + sum_pN, where N = number of assets        
    M.addConstr(SM[-1] >= Vects)
         
    M.setParam('TimeLimit',TL2)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize()
    Xd = []
    for j in range(Nlarge):
        Xd.append(xv[j].x)
    return M, M.ObjVal, xv, SM, Xd



def AddWarmStart(M, assets_biogas, xv, WS, N_initial_portfolio, N_entire_portfolio):
    KEYS = list(assets_biogas.keys())
    for j in range(N_entire_portfolio):
        ASSET = KEYS[j]
        strts  = assets_biogas[ASSET]['SCH']
        
        if j<=N_initial_portfolio - 1:
             strts2 = WS[j]
             xv[j].start = np.array(strts2)
        else:
             xv[j].start = np.array(strts)  
    return M, xv

def GI_WS1(Nlarge, TL1, TL2, InitN, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    SM = []
    xv = [[]]*Nlarge
    OBJ = 0
    for j in range(InitN):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
         
         
        
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(Nlarge)+' assets in the pool. Asset '+str(j)+' is added to the model')
    # variable SM[-1] is the sum of sum_p1 + sum_p2 + ... + sum_pN, where N = number of assets        
    M.addConstr(SM[-1] >= Vects)         
    M.setParam('TimeLimit',TL1)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize()
    M.remove(M.getConstrs()[-T:])
    
    for j in range(InitN, Nlarge):
        tT = AINQ[j].shape[1]
        
        xv[j] = M.addMVar(tT, lb = 0, ub = BOX[j])
        xv[j][IntIndx[j]].vType = gp.GRB.BINARY
        
        M.addConstr(AEQ[j]   @ xv[j]   ==   BEQ[j])
        M.addConstr(AINQ[j]  @ xv[j]   >=   BINQ[j] ) 
         
         
        
        if j == 0:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - CPLMTRX[j]@xv[j] == 0)
        else:
            SM.append(M.addMVar(T, lb=0, ub = 10**9))
            M.addConstr(SM[j] - SM[j-1] - CPLMTRX[j]@xv[j] == 0)
        OBJ = OBJ + OBJFN[j] @ xv[j]
        print(str(Nlarge)+' assets in the pool. Asset '+str(j)+' is added to the model')
    # variable SM[-1] is the sum of sum_p1 + sum_p2 + ... + sum_pN, where N = number of assets        
    M.addConstr(SM[-1] >= Vects)         
    M.setParam('TimeLimit',TL2)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize()
   
    
    return M, M.ObjVal, xv, SM

def PrepareMatrixInputs(portfolio_of_biogas_assets, var_input, env):
    run_optimization    = 0
    asset_index         = 0
        
    AINQ       =[[]]*len(portfolio_of_biogas_assets)
    BINQ       =[[]]*len(portfolio_of_biogas_assets)
    AEQ        =[[]]*len(portfolio_of_biogas_assets)
    BEQ        =[[]]*len(portfolio_of_biogas_assets)
    OBJFN      =[[]]*len(portfolio_of_biogas_assets)
    CPLMTRX    =[[]]*len(portfolio_of_biogas_assets)
    IntIndx    =[[]]*len(portfolio_of_biogas_assets)
    BOX        =[[]]*len(portfolio_of_biogas_assets)
    
    KEYS = list(portfolio_of_biogas_assets.keys())
    
    while asset_index < len(portfolio_of_biogas_assets.keys()):
        ASSET = KEYS[asset_index]
        asset_optimized = portfolio_of_biogas_assets[ASSET]
        M, AINQ[asset_index], BINQ[asset_index], AEQ[asset_index], BEQ[asset_index], OBJFN[asset_index], CPLMTRX[asset_index], BOX[asset_index], IntIndx[asset_index] = ModelSinglePP_BiogasMTRX(var_input, asset_optimized, run_optimization, [], env)
        asset_index = asset_index + 1
        print('Input matrices for Asset '+PrintNumb(asset_index)+' are produced')
    return AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx

def ModelSinglePP_Biogas(var_input, asset_optimized, run_optimization, M, env):
    Frc = var_input[0]     # price vector
    L01  = var_input[2]    # initial storage 
    vAdd = var_input[3]    # states of switch-ons 
    wAdd = var_input[4]    # states of switch-offs
     
    T = len(Frc)                        #the length of the price vector. Horizon
    idd     = np.eye(T)                 #idd is an identity matrix.
    l_tr    = np.tril(np.ones([T,T]),0) #l_tr is a lower triangular matrix
    zeroVec = np.array([0]*T)           #vector of zeros. Used for defining timing constraints
    onesVec = np.array([1]*T)           #vector of ones.  Used for defining timing constraints
     
    Capacity = asset_optimized['TechnicalFeature']['Capacity']
    Inflows  = np.ones([1, T])*asset_optimized['TechnicalFeature']['GasInflow']
    storage_rhs = np.cumsum(Inflows)
    storage_rhs = L01 + np.array(storage_rhs)  
    # This function can take an existing model and update it. 
    # If M is empty, then the model will be defined within this function.
    if not M:
        if not env:
            M = Model()
        else:
            M = Model(env=env)
    
    power_min   = []
    power_max   = []
    penalty_on  = []
    penalty_off = []
    UT_turbine    = []
    DT_turbine    = []
    
    p      = []
    u      = []
    v      = []
    w      = []
    assets_index     = 0
    for tb in asset_optimized['TechnicalFeature']['Turbines'].keys():
        """ Getting the constants related to the turbines:
        power_min, power_max, Uptime, Downtime, Penalties for switch on/ and ff"""
        power_min.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['Pmin'])
        power_max.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['Pmax'])
        penalty_on.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['StartCost'])
        penalty_off.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['StopCost'])
        UT_turbine.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['TminOn'])
        DT_turbine.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['TminOff'])
 
        # variables u,v,w,p
        u.append(M.addMVar(T, 0, 1))
        v.append(M.addMVar(T, 0, 1))
        w.append(M.addMVar(T, 0, 1))
        p.append(M.addMVar(T, 0, asset_optimized['TechnicalFeature']['Turbines'][tb]['Pmax']))
        u[assets_index].vType = gp.GRB.BINARY
        #u,v,w are all binary. It is enough to define u to be binary
        #if u is binary, then v and w are also binary
        M.addConstr(idd@u[assets_index] - l_tr@v[assets_index] + l_tr@w[assets_index] == 0)
        # Equation u(t)-u(t-1) = v(t) - w(t). Eq (8).
        M.addConstr(idd@p[assets_index] - power_min[assets_index]*idd@u[assets_index] >= 0)
        M.addConstr(power_max[assets_index]*idd@u[assets_index] - idd@p[assets_index] >= 0)
        # Equations p(t) - power_min*u(t) >=0 and power_max*u(t)-p(t)>=0. Eq (4).
        
        A2       = np.tril(np.ones([T,T]),-UT_turbine[assets_index])
        BU       = l_tr - A2 
        A2       = np.tril(np.ones([T,T]),-DT_turbine[assets_index])
        BT       = l_tr - A2 
        
        M.addConstr(idd@u[assets_index] + BT @ w[assets_index] <= onesVec-wAdd) 
        M.addConstr(idd@u[assets_index] - BU @ v[assets_index] >= zeroVec+vAdd)
        # Timing Constraint. Minimum up/down-time. Eq (9).
        assets_index = assets_index + 1
    
    Smp = M.addMVar(T, 0, sum(power_max))
    M.addConstr(Smp - sum(p)   == 0)
    #Smp is the sum over all vectors p(i). Smp = p1 + p2 + ... + p_N
    Storage  = M.addMVar(T, 0, Capacity)
    Spill  = M.addMVar(T, 0, Capacity)
    
    M.addConstr(idd @ Storage + l_tr @ Smp + l_tr @ Spill == storage_rhs)
    # Water Balance Equation Eq (6)
    
    """It is allowed to decide if model M defined up to now will be optimized.
    If so, the objective is defined according to Eq (1). But apart from this,
    a penalty for the spill is introduce to make sure that the spill of the 
    fuel will not take place"""
    if run_optimization:        
        OBJ = np.array(Frc) @  Smp  
        for assets_index in range(len(asset_optimized['TechnicalFeature']['Turbines'].keys())):
            OBJ = OBJ - penalty_on[assets_index]*np.ones(T) @ v[assets_index] - penalty_off[assets_index]*np.ones(T) @ w[assets_index] 
        OBJ = OBJ - 10*np.ones(T) @ Spill
        M.setParam('TimeLimit',60)
        M.setParam('OutputFlag',0)
        """ The objective OBJ @ [0*u,0*v,0*w,0*p,Smp,0*Storage,Spill] is maximized
        A @ b denotes the dot product A and b."""
        M.setObjective(OBJ, GRB.MAXIMIZE)
        M.optimize() 
        return M, Smp, p, u, v, w, Storage, Spill
    else:
        """Even if Optimization is not conducted, the output model M can be filled with 
        new variables and constraints and be optimized later. 
        If M was optimized, it is still possible to add new variables and to add new constraints
        or remove old ones. When M is reoptimized, the results of the previous optimizations can
        in many cases be the warm start of the enriched M"""
        return M, Smp, p, u, v, w, Storage, Spill
    
    
""""The following function solves the same problem as ModelSinglePP_Biogas, but in a different manner.
The problem is represented as follows:
    max p@x  s.t. A@x = a, B@x >= b, x[i] in {0,1}
    
    For the case of two turbines, the variables x is:
        x = [u1,v1,w1,p1,u2,v2,w2,p2,Smp,Storage,Spill]. """
def ModelSinglePP_BiogasMTRX(var_input, asset_optimized, run_optimization, M, env):
    Frc = var_input[0]
    L01  = var_input[2] 
    vAdd = var_input[3]
    wAdd = var_input[4]
     
    Frc = Frc
    T = len(Frc)
    idd      = np.eye(T)
    l_tr       = np.tril(np.ones([T,T]),0)
    onesVec         = np.array([1]*T)
     
    Capacity = asset_optimized['TechnicalFeature']['Capacity']
    Inflows  = np.ones([1, T])*asset_optimized['TechnicalFeature']['GasInflow']
    storage_rhs = np.cumsum(Inflows)
    storage_rhs = L01 + np.array(storage_rhs)    
    if not M:
        if not env:
            M = Model()
        else:
            M = Model(env=env)
    
    power_min   = []
    power_max   = []
    penalty_on  = []
    penalty_off = []
    UT_turbine    = []
    DT_turbine    = []
    
    assets_index     = 0
    
    NTurb = len(asset_optimized['TechnicalFeature']['Turbines'])

    LEN = T*(4*NTurb + 3)
    UVW = []           
    LHS_power_minC = []     
    LHS_power_maxC = []     
    LHS_UT = []         
    LHS_DT = []        


    for tb in asset_optimized['TechnicalFeature']['Turbines'].keys():
        power_min.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['Pmin'])
        power_max.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['Pmax'])
        penalty_on.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['StartCost'])
        penalty_off.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['StopCost'])
        UT_turbine.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['TminOn'])
        DT_turbine.append(asset_optimized['TechnicalFeature']['Turbines'][tb]['TminOff'])

        UINP = HCONCAT([idd, -l_tr, l_tr,0*idd])
        if assets_index == 0:
            UVW.append(HCONCAT([UINP, np.zeros([UINP.shape[0], LEN - UINP.shape[1]])]))
        else:
            UVW.append(HCONCAT([np.zeros([UINP.shape[0], 4*T*assets_index]), UINP, np.zeros([UINP.shape[0], LEN - UINP.shape[1] - 4*T*assets_index])]))
                         
        UINP_min = HCONCAT([-power_min[assets_index]*idd, 0*l_tr, 0*l_tr,  idd])
        UINP_max = HCONCAT([power_max[assets_index]*idd,  0*l_tr, 0*l_tr, -idd])
        if assets_index == 0:
            LHS_power_minC.append(HCONCAT([UINP_min, np.zeros([UINP_min.shape[0], LEN - UINP_min.shape[1]])]))
            LHS_power_maxC.append(HCONCAT([UINP_max, np.zeros([UINP_max.shape[0], LEN - UINP_max.shape[1]])]))
        else:
            LHS_power_minC.append(HCONCAT([np.zeros([UINP_min.shape[0], 4*T*assets_index]), UINP_min, np.zeros([UINP_min.shape[0], LEN - UINP_min.shape[1] - 4*T*assets_index])]))
            LHS_power_maxC.append(HCONCAT([np.zeros([UINP_max.shape[0], 4*T*assets_index]), UINP_max, np.zeros([UINP_max.shape[0], LEN - UINP_max.shape[1] - 4*T*assets_index])]))
                
        A2       = np.tril(np.ones([T,T]),-UT_turbine[assets_index])
        BU       = l_tr - A2 
        A2       = np.tril(np.ones([T,T]),-DT_turbine[assets_index])
        BT       = l_tr - A2 
     
        UINP = HCONCAT([idd, -BU, 0*l_tr,  0*idd]) #v

        if assets_index == 0:
            LHS_UT.append(HCONCAT([UINP, np.zeros([UINP.shape[0], LEN - UINP.shape[1]])]))
        else:
            LHS_UT.append(HCONCAT([np.zeros([UINP.shape[0], 4*T*assets_index]), UINP, np.zeros([UINP.shape[0], LEN - UINP.shape[1] - 4*T*assets_index])]))
       
        UINP = HCONCAT([idd, 0*l_tr, BT,  0*idd])  #w
        if assets_index == 0:
            LHS_DT.append(HCONCAT([UINP, np.zeros([UINP.shape[0], LEN - UINP.shape[1]])]))
        else:
            LHS_DT.append(HCONCAT([np.zeros([UINP.shape[0], 4*T*assets_index]), UINP, np.zeros([UINP.shape[0], LEN - UINP.shape[1] - 4*T*assets_index])]))
                        
        assets_index = assets_index + 1
    
    UVW           = VCONCAT(UVW)
    LHS_power_minC     = VCONCAT(LHS_power_minC)
    LHS_power_maxC     = VCONCAT(LHS_power_maxC)
    LHS_UT        = VCONCAT(LHS_UT)
    LHS_DT        = VCONCAT(LHS_DT)
    
    UVW_rhs       = np.zeros(UVW.shape[0])
    Ainq = VCONCAT([LHS_power_minC, LHS_power_maxC, LHS_UT, -LHS_DT])
    binq = VCONCAT([np.zeros(LHS_power_minC.shape[0]), np.zeros(LHS_power_maxC.shape[0]), VCONCAT([vAdd]*NTurb), VCONCAT([wAdd-onesVec]*NTurb)])
    
    UINP = HCONCAT([idd*0, idd*0, idd*0,  -idd])
    UINP = [UINP]*NTurb
    AGGR = [HCONCAT(UINP), idd, 0*idd, 0*idd]
    AGGR = HCONCAT(AGGR)
    aggrRHS = np.zeros(T)  #==
    
    WBE  = [HCONCAT([idd*0]*4*NTurb), l_tr, idd, l_tr]
    
    WBE = HCONCAT(WBE)
    wbeRH = storage_rhs  # ==
    
    Aeq = VCONCAT([UVW, AGGR, WBE])
    
    beq = VCONCAT([UVW_rhs, aggrRHS, wbeRH])
    
    OBJf = []
    for assets_index in range(len(penalty_on)):
        OBJf.append(VCONCAT([np.zeros(T), -np.ones(T)*penalty_on[assets_index], -np.ones(T)*penalty_off[assets_index],np.zeros(T)]))
    OBJf = VCONCAT(OBJf)
    OBJf = VCONCAT([OBJf, Frc])
    OBJf = VCONCAT([OBJf, np.zeros(T)])
    OBJf = VCONCAT([OBJf, -10*np.ones(T)])
    
    Box = []
    for assets_index in range(len(penalty_on)):
        Box.append(VCONCAT([np.ones(T), np.ones(T), np.ones(T),np.ones(T)*power_max[assets_index]]))
    Box = VCONCAT(Box)
    Box = VCONCAT([Box,np.ones(T)*sum(power_max)])
    Box = VCONCAT([Box,np.ones(T)*Capacity])
    Box = VCONCAT([Box,np.ones(T)*max(power_max)])
    
    INTGR = range(T)
    if NTurb>1:
        for j in range(NTurb-1):
            INTGR = list(INTGR) + list(  np.array(range(T)) + 4*T*(j+1) )
            
    COUPLINGMatr = [0*idd]*4*NTurb
    COUPLINGMatr = HCONCAT(COUPLINGMatr)
    COUPLINGMatr = HCONCAT([COUPLINGMatr, idd, idd*0, idd*0])
    strs = timer()
    M = []
    if not M:
        if not env:
            M = Model()
        else:
            M = Model(env=env)
       
    tT = Ainq.shape[1]
    
    xv = M.addMVar(tT, lb = 0, ub = Box)
    xv[INTGR].vType = gp.GRB.BINARY
    
    M.addConstr(Aeq   @ xv   ==   beq)
    M.addConstr(Ainq  @ xv   >=  binq)    
            
    OBJ = OBJf @ xv
    M.setParam('OutputFlag',0)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    if run_optimization:
        M.optimize()
        ends = timer()
        CTIME0 = ends - strs
        u       = []
        v       = []
        w       = []
        p       = []
        Smp     = []
        Storage = []
        Spill   = []
        for tb in range(NTurb):
            u.append( xv[np.array(range(T))+tb*4*T].x  )
            v.append( xv[np.array(range(T))+1*T+tb*4*T].x  )
            w.append( xv[np.array(range(T))+2*T+tb*4*T].x  )
            p.append( xv[np.array(range(T))+3*T+tb*4*T].x  )
        Smp = xv[ np.array(range(T)) + 4*NTurb*T ].x
        Storage = xv[ np.array(range(T)) + 4*NTurb*T + T ].x
        Spill = xv[ np.array(range(T)) + 4*NTurb*T + T ].x
        return M, Ainq, binq, Aeq, beq, OBJf, COUPLINGMatr, Box, u,v,w,p,Smp,Storage,Spill, INTGR, CTIME0
    else:
        return M, Ainq, binq, Aeq, beq, OBJf, COUPLINGMatr, Box, INTGR

""" The following function extracts the u,v,w,p,Smp,Storage, and Spill component
of vector xv. This function processes the xv[j] output of functions GI_WS and GI_Scratch"""

def ExtractValues(xv, T, NTurb):
    u       = []
    v       = []
    w       = []
    p       = []
    Smp     = []
    Storage = []
    Spill   = []
    for tb in range(NTurb):
        u.append( xv[np.array(range(T))+tb*4*T].x  )
        v.append( xv[np.array(range(T))+1*T+tb*4*T].x  )
        w.append( xv[np.array(range(T))+2*T+tb*4*T].x  )
        p.append( xv[np.array(range(T))+3*T+tb*4*T].x  )
    Smp = xv[ np.array(range(T)) + 4*NTurb*T ].x
    Storage = xv[ np.array(range(T)) + 4*NTurb*T + T ].x
    Spill = xv[ np.array(range(T)) + 4*NTurb*T + 2*T ].x
    return u,v,w,p,Smp,Storage,Spill

# Visualization function
def Visualize(VECTOR_1, VECTOR_2, n1, n2, Y_labelSum, Y_labelPrice, TITLE, m):
    fig, ax1 = plt.subplots()
    n1 = 0
    n2 = len(VECTOR_2)
    color = 'tab:red'
    ax1.set_xlabel('time (Hours)')
    ax1.set_ylabel(Y_labelSum, color=color)
    ax1.plot(VECTOR_1[(n1-m):(n2-m)], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.2*max(VECTOR_1)])
    ax1.set_title(TITLE)
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(Y_labelPrice, color=color) # we already handled the x-label with ax1
    ax2.plot(VECTOR_2[n1:n2], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.show() 
 
def PrintNumb(asset_index):
    if asset_index<10:
        return ' '+str(asset_index)
    else:
        return str(asset_index)

def HSTACK(Matrixes):
        outP = Matrixes[0]
        if len(Matrixes)>1:
            for j in range(1, len(Matrixes)):
                outP = hstack([outP, coo_matrix(Matrixes[j])])
        return outP

def VSTACK(Matrixes):
        outP = HSTACK(Matrixes[0])
        if len(Matrixes)>1:
            for j in range(1, len(Matrixes)):
                outP = vstack([outP, HSTACK(Matrixes[j])])
        return outP

def GI_ScratchDirectSum(InitN, TL, Vects, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env):
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    LEN = 0
    INTindx = []
    for j in range(len(OBJFN)):
        LEN = LEN + len(OBJFN[j])
        ASSET = KEYS[j]
        strts  = assets_biogas[ASSET]['SCH']
        INTindx0 = 0*np.array(strts)
        INTindx0[IntIndx[j]] = j+1
        if j==0:
            print(str(InitN)+' assets in the pool. Asset '+str(j)+' is added to the model')
            INTindx    = INTindx0
            OBJCTV     = OBJFN[j]
            COUPL_MATR = coo_matrix(CPLMTRX[j])
            Aeq        = coo_matrix(AEQ[j])
            Ainq       = coo_matrix(AINQ[j])
            BoxC       = BOX[j]
            beq        = BEQ[j]
            binq       = BINQ[j]
        else:
            print(str(InitN)+' assets in the pool. Asset '+str(j)+' is added to the model')
            INTindx    = VCONCAT([INTindx, INTindx0])
            OBJCTV     = VCONCAT([OBJCTV,  OBJFN[j]])
            COUPL_MATR = HSTACK([COUPL_MATR, coo_matrix(CPLMTRX[j])])
            Aeq        = scipy.sparse.block_diag((Aeq   ,  coo_matrix(AEQ[j])) )
            Ainq       = scipy.sparse.block_diag((Ainq  ,  coo_matrix(AINQ[j])))
            BoxC       = VCONCAT([BoxC, BOX[j]])
            beq        = VCONCAT([beq, BEQ[j]])
            binq       = VCONCAT([binq, BINQ[j]])
    INTindex = INTindx.nonzero()
    M = []
    if not env:
        M = Model()
    else:
        M = Model(env=env)
    tT = len(OBJCTV)
    
    xv = M.addMVar(tT, lb = 0, ub = BoxC)
    xv[INTindex].vType = gp.GRB.BINARY
    
    M.addConstr(Aeq          @ xv   ==   beq  )
    M.addConstr(Ainq         @ xv   >=   binq ) 
    M.addConstr(COUPL_MATR   @ xv   >=   Vects)
    
    OBJ = OBJCTV @ xv
         
    M.setParam('TimeLimit',TL)
    M.setObjective(OBJ, GRB.MAXIMIZE)
    M.optimize() 
    
    return M.ObjVal, M


def positiv(vector):
    vect = []
    for v in vector:
        if v >= 0:
            vect.append(v)
        else:
            vect.append(0)
    return np.array(vect)

def negativ(vector):
    vect = []
    for v in vector:
        if v <= 0:
            vect.append(-v)
        else:
            vect.append(0)
    return np.array(vect)
