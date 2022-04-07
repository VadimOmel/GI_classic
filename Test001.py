# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:43:23 2022

@author: OMELCHENKO_V
"""

WS, M0   =      GI_Scratch(7, 135, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1, WS =  GI_WS2(10, 135, WS,    7, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

M1, ObjV, xv1, SM1, WS =  GI_WS2(15, 135, WS,    10, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

M1, ObjV, xv1, SM1, WS =  GI_WS2(25, 135, WS,    15, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

M1, ObjV, xv1, SM1, WS =  GI_WS2(35, 135, WS,    25, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
M1.setParam('TimeLimit',600)
M1.optimize()
M1.setParam('TimeLimit',2325)
M1.optimize()

#Result: Best objective 2.984337017337e+08, best bound 2.984873155367e+08, gap 0.0180%

WS, M0   =      GI_Scratch(7, 135, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
for j in range(7, 35):
    M1, ObjV, xv1, SM1, WS =  GI_WS2(j+1, 40+2*j, WS,   j, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

#Result: Best objective 2.984200384924e+08, best bound 2.985096958892e+08, gap 0.0300%  
WS, M0   =      GI_Scratch(35, 3600, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
WS, M0   =      GI_Scratch(35, 1600, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# Result: 
#1600 sec:  Best objective 2.984283440645e+08, best bound 2.984907828070e+08, gap 0.0209% (RealGap = 0.0197%)

 
 
 
WS, M0   =      GI_Scratch(7, 100, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1, WS =  GI_WS2(25, 100, WS,    7,  RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
M1, ObjV, xv1, SM1, WS =  GI_WS2(35, 400, WS,    25, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# Result: Best objective 2.984318845872e+08, best bound 2.984967147398e+08, gap 0.0217% (RealGap = 0.01857%)

WS, M0   =      GI_Scratch(7, 60, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1, WS =  GI_WS2(25, 60, WS,    7,  RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
M1, ObjV, xv1, SM1, WS =  GI_WS2(35, 480, WS,    25, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# Best objective 2.984300294924e+08, best bound 2.984971841951e+08, gap 0.0225%


WS, M0   =      GI_Scratch(7, 60, RHS_cpl,                             AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, env)
# WS is the output of GI_Scratch. 
# It is used as the warm start:
M1, ObjV, xv1, SM1, WS =  GI_WS2(14, 60, WS,    7,  RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

M1, ObjV, xv1, SM1, WS =  GI_WS2(25, 60, WS,    14,  RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)

M1, ObjV, xv1, SM1, WS =  GI_WS2(35, 420, WS,    25, RHS_cpl, AINQ, BINQ, AEQ, BEQ, OBJFN, CPLMTRX, BOX, IntIndx, T, KEYS, assets_biogas, env)
# Best objective 2.984244660615e+08, best bound 2.984972421644e+08, gap 0.0244% 