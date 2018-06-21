#!/usr/bin/env python

#------------------------------------------------------------------------------
# imsrg_pairing.py
#
# author:   H. Hergert 
# version:  1.5.0
# date:     Dec 6, 2016
# 
# tested with Python v2.7
# 
# Solves the pairing model for four particles in a basis of four doubly 
# degenerate states by means of an In-Medium Similarity Renormalization 
# Group (IMSRG) flow.
#
#------------------------------------------------------------------------------

import numpy as np
from numpy import array, dot, diag, reshape, transpose
from scipy.linalg import eigvalsh
from scipy.integrate import odeint, ode

import time

from sys import argv

#-----------------------------------------------------------------------------------
# basis and index functions
#-----------------------------------------------------------------------------------

def construct_basis_2B(holes, particles):
  basis = []
  for i in holes:
    for j in holes:
      basis.append((i, j))

  for i in holes:
    for a in particles:
      basis.append((i, a))

  for a in particles:
    for i in holes:
      basis.append((a, i))

  for a in particles:
    for b in particles:
      basis.append((a, b))

  return basis

# how is this different from construct_basis_2B ???
def construct_basis_ph2B(holes, particles):
  basis = []
  for i in holes:
    for j in holes:
      basis.append((i, j))

  for i in holes:
    for a in particles:
      basis.append((i, a))

  for a in particles:
    for i in holes:
      basis.append((a, i))

  for a in particles:
    for b in particles:
      basis.append((a, b))

  return basis


#
# We use dictionaries for the reverse lookup of state indices
#
def construct_index_2B(bas2B):
  index = { }
  for i, state in enumerate(bas2B):
    index[state] = i

  return index


#SRS
#SRS  3-body stuff
#SRS

# Doing this about as dumb as can be

def construct_basis_3B(holes, particles):
  basis = []
  for i in holes:
    for j in holes:
      for k in holes:
        basis.append((i, j, k))

  for i in holes:
    for j in holes: 
      for a in particles:
        basis.append((i, j, a))
        basis.append((i, a, j))
        basis.append((a, i, j))

  for i in holes:
    for a in particles:
      for b in particles:
        basis.append((i, a, b))
        basis.append((a, i, b))
        basis.append((a, b, i))

  for a in particles:
    for b in particles:
      for c in particles:
        basis.append((a, b, c))

  return basis


#SRS this is identical to construct_index_2B
def construct_index_3B(bas3B):
  index = { }
  for i, state in enumerate(bas3B):
    index[state] = i

  return index

#-----------------------------------------------------------------------------------
# transform matrices to particle-hole representation
#-----------------------------------------------------------------------------------
def ph_transform_2B(Gamma, bas2B, idx2B, basph2B, idxph2B):
  dim = len(basph2B)
  Gamma_ph = np.zeros((dim, dim))

  for i1, (a,b) in enumerate(basph2B):
    for i2, (c, d) in enumerate(basph2B):
      Gamma_ph[i1, i2] -= Gamma[idx2B[(a,d)], idx2B[(c,b)]]

  return Gamma_ph

def inverse_ph_transform_2B(Gamma_ph, bas2B, idx2B, basph2B, idxph2B):
  dim = len(bas2B)
  Gamma = np.zeros((dim, dim))

  for i1, (a,b) in enumerate(bas2B):
    for i2, (c, d) in enumerate(bas2B):
      Gamma[i1, i2] -= Gamma_ph[idxph2B[(a,d)], idxph2B[(c,b)]]
  
  return Gamma

#-----------------------------------------------------------------------------------
# commutator of matrices
#-----------------------------------------------------------------------------------
def commutator(a,b):
  return dot(a,b) - dot(b,a)

#-----------------------------------------------------------------------------------
# norms of off-diagonal Hamiltonian pieces
#-----------------------------------------------------------------------------------
def calc_fod_norm(f, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  
  norm = 0.0
  for a in particles:
    for i in holes:
      norm += f[a,i]**2 + f[i,a]**2

  return np.sqrt(norm)

def calc_Gammaod_norm(Gamma, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  norm = 0.0
  for a in particles:    
    for b in particles:
      for i in holes:
        for j in holes:
          norm += Gamma[idx2B[(a,b)],idx2B[(i,j)]]**2 + Gamma[idx2B[(i,j)],idx2B[(a,b)]]**2

  return np.sqrt(norm)



#SRS
def calc_Wod_norm(W, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx3B     = user_data["idx3B"]

  norm = 0.0
  for a in particles:    
    for b in particles:
      for c in particles:
        for i in holes:
          for j in holes:
            for k in holes:
              norm += W[idx3B[(a,b,c)],idx3B[(i,j,k)]]**2 + W[idx3B[(i,j,k)],idx3B[(a,b,c)]]**2

  return np.sqrt(norm)



#SRS
def calc_Whhp_norm(W, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx3B     = user_data["idx3B"]

  norm = 0.0
  for a in particles:    
    for b in particles:
      for i in holes:
        for j in holes:
          for k in holes:
            for l in holes:
              norm += W[idx3B[(a,i,j)],idx3B[(b,k,l)]]**2 + W[idx3B[(b,k,l)],idx3B[(a,i,j)]]**2

  return np.sqrt(norm)


#-----------------------------------------------------------------------------------
# occupation number matrices
#-----------------------------------------------------------------------------------
def construct_occupation_1B(bas1B, holes, particles):
  dim = len(bas1B)
  occ = np.zeros(dim)

  for i in holes:
    occ[i] = 1.

  return occ

# diagonal matrix: n_a - n_b
def construct_occupationA_2B(bas2B, occ1B):
  dim = len(bas2B)
  occ = np.zeros((dim,dim))

  for i1, (i,j) in enumerate(bas2B):
    occ[i1, i1] = occ1B[i] - occ1B[j]

  return occ


# diagonal matrix: 1 - n_a - n_b
def construct_occupationB_2B(bas2B, occ1B):
  dim = len(bas2B)
  occ = np.zeros((dim,dim))

  for i1, (i,j) in enumerate(bas2B):
    occ[i1, i1] = 1. - occ1B[i] - occ1B[j]

  return occ

# diagonal matrix: n_a * n_b
def construct_occupationC_2B(bas2B, occ1B):
  dim = len(bas2B)
  occ = np.zeros((dim,dim))

  for i1, (i,j) in enumerate(bas2B):
    occ[i1, i1] = occ1B[i] * occ1B[j]

  return occ

#-----------------------------------------------------------------------------------
# generators
#-----------------------------------------------------------------------------------
def eta_brillouin(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      # (1-n_a)n_i - n_a(1-n_i) = n_i - n_a
      eta1B[a, i] =  f[a,i]
      eta1B[i, a] = -f[a,i]

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]]

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B

def eta_imtime(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      dE = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = np.sign(dE)*f[a,i]
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          dE = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = np.sign(dE)*Gamma[idx2B[(a,b)], idx2B[(i,j)]]

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_white(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_white_mp(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
          )

          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B

def eta_white_atan(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = 0.5 * np.arctan(2 * f[a,i]/denom)
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j] 
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]] 
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = 0.5 * np.arctan(2 * Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom)

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_wegner(f, Gamma, user_data):

  dim1B     = user_data["dim1B"]
  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas2B     = user_data["bas2B"]
  basph2B   = user_data["basph2B"]
  idx2B     = user_data["idx2B"]
  idxph2B   = user_data["idxph2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]


  # split Hamiltonian in diagonal and off-diagonal parts
  fd      = np.zeros_like(f)
  fod     = np.zeros_like(f)
  Gammad  = np.zeros_like(Gamma)
  Gammaod = np.zeros_like(Gamma)

  for a in particles:
    for i in holes:
      fod[a, i] = f[a,i]
      fod[i, a] = f[i,a]
  fd = f - fod

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          Gammaod[idx2B[(a,b)], idx2B[(i,j)]] = Gamma[idx2B[(a,b)], idx2B[(i,j)]]
          Gammaod[idx2B[(i,j)], idx2B[(a,b)]] = Gamma[idx2B[(i,j)], idx2B[(a,b)]]
  Gammad = Gamma - Gammaod


  #############################        
  # one-body flow equation  
  eta1B  = np.zeros_like(f)

  # 1B - 1B
  eta1B += commutator(fd, fod)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        for a in particles:
          eta1B[p,q] += (
            fd[i,a]  * Gammaod[idx2B[(a, p)], idx2B[(i, q)]] 
            - fd[a,i]  * Gammaod[idx2B[(i, p)], idx2B[(a, q)]] 
            - fod[i,a] * Gammad[idx2B[(a, p)], idx2B[(i, q)]] 
            + fod[a,i] * Gammad[idx2B[(i, p)], idx2B[(a, q)]]
          )

  # 2B - 2B
  # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
  GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        eta1B[p,q] += (
          0.5*GammaGamma[idx2B[(i,p)], idx2B[(i,q)]] 
          - transpose(GammaGamma)[idx2B[(i,p)], idx2B[(i,q)]]
        )

  GammaGamma = dot(Gammad, dot(occC_2B, Gammaod))
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        eta1B[p,q] += (
          0.5*GammaGamma[idx2B[(r,p)], idx2B[(r,q)]] 
          + transpose(GammaGamma)[idx2B[(r,p)], idx2B[(r,q)]] 
        )


  #############################        
  # two-body flow equation  
  eta2B = np.zeros_like(Gamma)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          for t in range(dim1B):
            eta2B[idx2B[(p,q)],idx2B[(r,s)]] += (
              fd[p,t] * Gammaod[idx2B[(t,q)],idx2B[(r,s)]] 
              + fd[q,t] * Gammaod[idx2B[(p,t)],idx2B[(r,s)]] 
              - fd[t,r] * Gammaod[idx2B[(p,q)],idx2B[(t,s)]] 
              - fd[t,s] * Gammaod[idx2B[(p,q)],idx2B[(r,t)]]
              - fod[p,t] * Gammad[idx2B[(t,q)],idx2B[(r,s)]] 
              - fod[q,t] * Gammad[idx2B[(p,t)],idx2B[(r,s)]] 
              + fod[t,r] * Gammad[idx2B[(p,q)],idx2B[(t,s)]] 
              + fod[t,s] * Gammad[idx2B[(p,q)],idx2B[(r,t)]]
            )

  
  # 2B - 2B - particle and hole ladders
  # Gammad.occB.Gammaod
  GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))

  eta2B += 0.5 * (GammaGamma - transpose(GammaGamma))

  # 2B - 2B - particle-hole chain
  
  # transform matrices to particle-hole representation and calculate 
  # Gammad_ph.occA_ph.Gammaod_ph
  Gammad_ph = ph_transform_2B(Gammad, bas2B, idx2B, basph2B, idxph2B)
  Gammaod_ph = ph_transform_2B(Gammaod, bas2B, idx2B, basph2B, idxph2B)

  GammaGamma_ph = dot(Gammad_ph, dot(occphA_2B, Gammaod_ph))

  # transform back to standard representation
  GammaGamma    = inverse_ph_transform_2B(GammaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

  # commutator / antisymmetrization
  work = np.zeros_like(GammaGamma)
  for i1, (i,j) in enumerate(bas2B):
    for i2, (k,l) in enumerate(bas2B):
      work[i1, i2] -= (
        GammaGamma[i1, i2] 
        - GammaGamma[idx2B[(j,i)], i2] 
        - GammaGamma[i1, idx2B[(l,k)]] 
        + GammaGamma[idx2B[(j,i)], idx2B[(l,k)]]
      )
  GammaGamma = work

  eta2B += GammaGamma


  return eta1B, eta2B


#-----------------------------------------------------------------

#SRS   3B  generator
#SRS
def eta_white_3B(f, Gamma, W, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]
  idx3B     = user_data["idx3B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  #SRS three-body part of the generator
  #SRS note that I negelect 3b contributions to the denominator
  eta3B = np.zeros_like(W)

  for a in particles:
    for b in particles:
      for c in particles:
        for i in holes:
          for j in holes:
            for k in holes:
              denom = ( 
                f[a,a] + f[b,b] +f[c,c] - f[i,i] - f[j,j] -f[k,k] 
                + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
                + Gamma[idx2B[(a,c)],idx2B[(a,c)]] 
                + Gamma[idx2B[(b,c)],idx2B[(b,c)]] 
                + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
                + Gamma[idx2B[(i,k)],idx2B[(i,k)]]
                + Gamma[idx2B[(j,k)],idx2B[(j,k)]]
                - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
                - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
                - Gamma[idx2B[(a,k)],idx2B[(a,k)]] 
                - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
                - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
                - Gamma[idx2B[(b,k)],idx2B[(b,k)]] 
                - Gamma[idx2B[(c,i)],idx2B[(c,i)]] 
                - Gamma[idx2B[(c,j)],idx2B[(c,j)]] 
                - Gamma[idx2B[(c,k)],idx2B[(c,k)]] 
              )
    
              val = W[idx3B[(a,b,c)], idx3B[(i,j,k)]] / denom
    
              eta3B[idx3B[(a,b,c)],idx3B[(i,j,k)]] = val
              eta3B[idx3B[(i,j,k)],idx3B[(a,b,c)]] = -val



  return eta1B, eta2B, eta3B







#-----------------------------------------------------------------------------------
# derivatives 
#-----------------------------------------------------------------------------------
def flow_imsrg2(eta1B, eta2B, f, Gamma, user_data):

  dim1B     = user_data["dim1B"]
  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  basph2B   = user_data["basph2B"]
  idxph2B   = user_data["idxph2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]

  #############################        
  # zero-body flow equation
  dE = 0.0

  for i in holes:
    for a in particles:
      dE += eta1B[i,a] * f[a,i] - eta1B[a,i] * f[i,a]

  for i in holes:
    for j in holes:
      for a in particles:
        for b in particles:
          dE += 0.5 * eta2B[idx2B[(i,j)], idx2B[(a,b)]] * Gamma[idx2B[(a,b)], idx2B[(i,j)]]


  #############################        
  # one-body flow equation  
  df  = np.zeros_like(f)

  # 1B - 1B
  df += commutator(eta1B, f)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        for a in particles:
          df[p,q] += (
            eta1B[i,a] * Gamma[idx2B[(a, p)], idx2B[(i, q)]] 
            - eta1B[a,i] * Gamma[idx2B[(i, p)], idx2B[(a, q)]] 
            - f[i,a] * eta2B[idx2B[(a, p)], idx2B[(i, q)]] 
            + f[a,i] * eta2B[idx2B[(i, p)], idx2B[(a, q)]]
          )

  # 2B - 2B
  # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
  etaGamma = dot(eta2B, dot(occB_2B, Gamma))
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        df[p,q] += 0.5*(
          etaGamma[idx2B[(i,p)], idx2B[(i,q)]] 
          + transpose(etaGamma)[idx2B[(i,p)], idx2B[(i,q)]]
        )

  etaGamma = dot(eta2B, dot(occC_2B, Gamma))
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        df[p,q] += 0.5*(
          etaGamma[idx2B[(r,p)], idx2B[(r,q)]] 
          + transpose(etaGamma)[idx2B[(r,p)], idx2B[(r,q)]] 
        )


  #############################        
  # two-body flow equation  
  dGamma = np.zeros_like(Gamma)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          for t in range(dim1B):
            dGamma[idx2B[(p,q)],idx2B[(r,s)]] += (
              eta1B[p,t] * Gamma[idx2B[(t,q)],idx2B[(r,s)]] 
              + eta1B[q,t] * Gamma[idx2B[(p,t)],idx2B[(r,s)]] 
              - eta1B[t,r] * Gamma[idx2B[(p,q)],idx2B[(t,s)]] 
              - eta1B[t,s] * Gamma[idx2B[(p,q)],idx2B[(r,t)]]
              - f[p,t] * eta2B[idx2B[(t,q)],idx2B[(r,s)]] 
              - f[q,t] * eta2B[idx2B[(p,t)],idx2B[(r,s)]] 
              + f[t,r] * eta2B[idx2B[(p,q)],idx2B[(t,s)]] 
              + f[t,s] * eta2B[idx2B[(p,q)],idx2B[(r,t)]]
            )

  
  # 2B - 2B - particle and hole ladders
  # eta2B.occB.Gamma
  etaGamma = dot(eta2B, dot(occB_2B, Gamma))

  dGamma += 0.5 * (etaGamma + transpose(etaGamma))

  # 2B - 2B - particle-hole chain
  
  # transform matrices to particle-hole representation and calculate 
  # eta2B_ph.occA_ph.Gamma_ph
  eta2B_ph = ph_transform_2B(eta2B, bas2B, idx2B, basph2B, idxph2B)
  Gamma_ph = ph_transform_2B(Gamma, bas2B, idx2B, basph2B, idxph2B)

  etaGamma_ph = dot(eta2B_ph, dot(occphA_2B, Gamma_ph))

  # transform back to standard representation
  etaGamma    = inverse_ph_transform_2B(etaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

  # commutator / antisymmetrization
  work = np.zeros_like(etaGamma)
  for i1, (i,j) in enumerate(bas2B):
    for i2, (k,l) in enumerate(bas2B):
      work[i1, i2] -= (
        etaGamma[i1, i2] 
        - etaGamma[idx2B[(j,i)], i2] 
        - etaGamma[i1, idx2B[(l,k)]] 
        + etaGamma[idx2B[(j,i)], idx2B[(l,k)]]
      )
  etaGamma = work

  dGamma += etaGamma


  return dE, df, dGamma

#-------------------------------------------------------------------------

#SRS 
def flow_imsrg3(eta1B, eta2B, eta3B, f, Gamma, W, user_data):
  dim1B     = user_data["dim1B"]
  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas2B     = user_data["bas2B"]
  bas3B     = user_data["bas3B"]
  idx2B     = user_data["idx2B"]
  basph2B   = user_data["basph2B"]
  idxph2B   = user_data["idxph2B"]
  idx3B     = user_data["idx3B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]

  #SRS Reuse the imsrg(2) stuf f
  dE, df, dGamma = flow_imsrg2 (eta1B, eta2B, f, Gamma, user_data)
  dW = np.zeros_like(W)        
                               
  dG3 = np.zeros_like(dGamma)
                               
  #SRS  BAIL OUT               
#  return dE, df, dGamma, dW   
                               
                               
  #SRS three body piece (double gulp)
  #SRS [2,2]->3
  tstart = time.time()
#  print "[2,2]->3"
  for bra,(p,q,r) in enumerate(bas3B):
    pq,rq,pr = idx2B[(p,q)], idx2B[(r,q)], idx2B[(p,r)]
    for ket,(s,t,u) in enumerate(bas3B):
      if ket<bra: continue
      tu,su,ts = idx2B[(t,u)], idx2B[(s,u)], idx2B[(t,s)]
      dw = 0.
      for v in range(dim1B):
        sv,tv,uv = idx2B[(s,v)], idx2B[(t,v)], idx2B[(u,v)]
        vp,vq,vr = idx2B[(v,p)], idx2B[(v,q)], idx2B[(v,r)]
        dw  +=( eta2B[pq, sv ] * Gamma[vr, tu ]   
              - eta2B[rq, sv ] * Gamma[vp, tu ]   
              - eta2B[pr, sv ] * Gamma[vq, tu ]   
              - eta2B[pq, tv ] * Gamma[vr, su ]   
              + eta2B[rq, tv ] * Gamma[vp, su ]   
              + eta2B[pr, tv ] * Gamma[vq, su ]   
              - eta2B[pq, uv ] * Gamma[vr, ts ]   
              + eta2B[rq, uv ] * Gamma[vp, ts ]   
              + eta2B[pr, uv ] * Gamma[vq, ts ]    )

        dw  -=(  Gamma[pq, sv ] * eta2B[vr, tu ]
              -  Gamma[rq, sv ] * eta2B[vp, tu ]
              -  Gamma[pr, sv ] * eta2B[vq, tu ]
              -  Gamma[pq, tv ] * eta2B[vr, su ]
              +  Gamma[rq, tv ] * eta2B[vp, su ]
              +  Gamma[pr, tv ] * eta2B[vq, su ]
              -  Gamma[pq, uv ] * eta2B[vr, ts ]
              +  Gamma[rq, uv ] * eta2B[vp, ts ]
              +  Gamma[pr, uv ] * eta2B[vq, ts ] )



#        if False and (p in particles) and (q in holes) and (r in holes) and (s in particles) and (t in holes) and (u in holes) and (p !=q and p!=r and q!=r and s!=t and s!=u and t!=u and s!=v and p!=v):
#          print '@  v = %d,   < %d %d %d | dW | %d %d %d > = eta_%d%d%d%d Gamma_%d%d%d%d =  %f x %f - %f x %f'%(v,p,q,r,s,t,u,q,r,s,v,v,p,t,u,
#                       eta2B[idx2B[(q,r)],idx2B[(s,v)]], Gamma[idx2B[(v,p)],idx2B[(t,u)]],
#                       Gamma[idx2B[(q,r)],idx2B[(s,v)]], eta2B[idx2B[(v,p)],idx2B[(t,u)]])

      dW[bra,ket] += dw
      dW[ket,bra] = dW[bra,ket]

  tstop = time.time()
#  print "elapsed: ",tstop-tstart

#  print "size of dG = ", np.linalg.norm(dG3)

#  print 'Norm eta3B = ', np.linalg.norm(eta3B)

#  print 'Norm of dW = ', np.linalg.norm(dW)

#  print '||dW_hhp|| = ', calc_Whhp_norm(dW, user_data)
#  if  np.linalg.norm(W) < 0.5: 
#    return dE, df, dGamma, dW


  dE3 = 0.
  #SRS zero body piece
#  print "[3,3]->0"
  tstart = time.time()
  for a in particles:
    for b in particles:
      for c in particles:
        for i in holes:
          for j in holes:
            for k in holes:
              dE3 += 1./18 * eta3B[idx3B[(i,j,k)], idx3B[(a,b,c)]] * W[idx3B[(a,b,c)], idx3B[(i,j,k)]]
  dE += dE3
#  print "dE3 = ",dE3
  tstop = time.time()
#  print "elapsed: ",tstop-tstart


  #SRS one body piece
  #SRS [3,3]->1
  # This can be written as a matrix multiplication with an occupation matrix thrown in there
#  print "[3,3]->1"
  tstart = time.time()
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        for j in holes:
          for a in particles:
            for b in particles:
              for c in particles:
                df[p,q] += 1./12 * eta3B[idx3B[(p,i,j)], idx3B[(a,b,c)]] *     W[idx3B[(a,b,c)], idx3B[(q,i,j)]]
                df[p,q] -= 1./12 *     W[idx3B[(p,i,j)], idx3B[(a,b,c)]] * eta3B[idx3B[(a,b,c)], idx3B[(q,i,j)]]
              for k in holes:
                df[p,q] += 1./12 * eta3B[idx3B[(p,a,b)], idx3B[(i,j,k)]] *     W[idx3B[(i,j,k)], idx3B[(q,a,b)]]
                df[p,q] -= 1./12 *     W[idx3B[(p,a,b)], idx3B[(i,j,k)]] * eta3B[idx3B[(i,j,k)], idx3B[(q,a,b)]]

  tstop = time.time()
#  print "elapsed: ",tstop-tstart

  #SRS [2,3]->1 and [3,2]->1
#  print "[2,3]->1"
  tstart = time.time()
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          for i in holes:
            for j in holes:
              df[p,q] += 0.25 * eta2B[idx2B[(i,j)], idx2B[(r,s)]] * (     W[idx3B[(r,s,p)], idx3B[(i,j,q)]] +     W[idx3B[(i,j,p)], idx3B[(r,s,q)]]  )
              df[p,q] -= 0.25 * Gamma[idx2B[(i,j)], idx2B[(r,s)]] * ( eta3B[idx3B[(r,s,p)], idx3B[(i,j,q)]] - eta3B[idx3B[(i,j,p)], idx3B[(r,s,q)]]  )

  tstop = time.time()
#  print "elapsed: ",tstop-tstart

  #SRS two body piece (gulp)
  #SRS [3,3]->2  ppph / hhph
  #SRS These should definitely be done with matrix multiplication...
#  print "[3,3]->2"
  tstart = time.time()
  if np.linalg.norm(eta3B)>1e-3:
    for bra, (p,q) in enumerate(bas2B):
      for ket, (r,s) in enumerate(bas2B):
            if ket < bra: continue
            dG = 0.
            for a in particles:
              pqa,rsa = idx3B[(p,q,a)],idx3B[(r,s,a)]
              for i in holes:
                pqi,rsi = idx3B[(p,q,i)],idx3B[(r,s,i)]
                for b in particles:
                  for c in particles:
                    abc = idx3B[(a,b,c)]
                    dG += 1./6 * ( eta3B[pqi,abc] * W[abc,rsi]  -  W[pqi,abc] * eta3B[abc,rsi] )
                for j in holes:
                  for k in holes:
                    ijk = idx3B[(i,j,k)]
                    dG -= 1./6 * ( eta3B[pqa,ijk] * W[ijk,rsa]  -   W[pqa,ijk] * eta3B[ijk,rsa] )
    #SRS [3,3]->2  phph
            for i in holes:
              for j in holes:
                pij,qij,rij,sij = idx3B[(p,i,j)], idx3B[(q,i,j)], idx3B[(r,i,j)], idx3B[(s,i,j)]
                for a in particles:
                  for b in particles:
                    abp,abq,abr,abS = idx3B[(a,b,p)], idx3B[(a,b,q)], idx3B[(a,b,r)], idx3B[(a,b,s)]
                    dG += 1./4 * ( eta3B[pij,abr]  *  W[abq,sij]
                                 - eta3B[qij,abr]  *  W[abp,sij]
                                 - eta3B[pij,abS]  *  W[abq,rij]
                                 + eta3B[qij,abS]  *  W[abp,rij]
                                 - eta3B[pij,rij]  *  W[abq,abS]
                                 + eta3B[qij,rij]  *  W[abp,abS]
                                 + eta3B[pij,sij]  *  W[abq,abr]
                                 - eta3B[qij,sij]  *  W[abp,abr]  )
            dG3[bra,ket] += dG
            dG3[ket,bra] = dG3[bra,ket]

  tstop = time.time()
#  print "elapsed: ",tstop-tstart

#  print "size of dG = ", np.linalg.norm(dG3)


  #SRS [2,3]->2 and [3,2]->2
#  print "[2,3]->2"
  tstart = time.time()
  for bra,(p,q) in enumerate( bas2B):
    for ket, (r,s) in enumerate(bas2B):
          if ket > bra : continue
          dG = 0.
          for i in holes:
            for a in particles:
              for b in particles:
                dG += 0.5 * ( eta2B[idx2B[(i,q)],idx2B[(a,b)]] *     W[idx3B[(p,a,b)],idx3B[(r,i,s)]]
                            - eta2B[idx2B[(i,p)],idx2B[(a,b)]] *     W[idx3B[(q,a,b)],idx3B[(r,i,s)]]
                            - eta2B[idx2B[(a,b)],idx2B[(i,s)]] *     W[idx3B[(p,i,q)],idx3B[(r,a,b)]]
                            + eta2B[idx2B[(a,b)],idx2B[(i,r)]] *     W[idx3B[(p,i,q)],idx3B[(s,a,b)]] )
#                if False and (p in particles) and (q in particles) and (r in holes) and (s in holes) and (p!=q) and (r!=s) and (p==5 and q==4 and r==3 and s==2):
#                  print ' (%d %d %d)'%(i,a,b)
#                  print '@  < %d %d | dG | %d %d > :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( p,q,r,s,i,q,a,b,p,a,b,r,i,s,
#                                                               eta2B[idx2B[(i,q)],idx2B[(a,b)]],  W[idx3B[(p,a,b)],idx3B[(r,i,s)]]  )
#                  print '@                         :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( i,p,a,b,q,a,b,r,i,s,
#                                                               eta2B[idx2B[(i,p)],idx2B[(a,b)]],  W[idx3B[(q,a,b)],idx3B[(r,i,s)]]  )
#                  print '@                         :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( a,b,i,s,p,i,q,r,a,b,
#                                                               eta2B[idx2B[(a,b)],idx2B[(i,s)]] , W[idx3B[(p,i,q)],idx3B[(r,a,b)]]  )
#                  print '@                         :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( a,b,i,r,p,i,q,s,a,b,
#                                                               eta2B[idx2B[(a,b)],idx2B[(i,r)]] , W[idx3B[(p,i,q)],idx3B[(s,a,b)]]  )
              for j in holes:
                dG += 0.5 * ( eta2B[idx2B[(a,q)],idx2B[(i,j)]] *     W[idx3B[(p,i,j)],idx3B[(r,a,s)]]
                            - eta2B[idx2B[(a,p)],idx2B[(i,j)]] *     W[idx3B[(q,i,j)],idx3B[(r,a,s)]]
                            - eta2B[idx2B[(i,j)],idx2B[(a,s)]] *     W[idx3B[(p,a,q)],idx3B[(r,i,j)]]
                            + eta2B[idx2B[(i,j)],idx2B[(a,r)]] *     W[idx3B[(p,a,q)],idx3B[(s,i,j)]] )
#                if True and (p in particles) and (q in particles) and (r in holes) and (s in holes) and (p!=q) and (r!=s) and (p==5 and q==4 and r==3 and s==2) and (i,a,j)==(3,4,2):
#                  print ' (%d %d %d)'%(i,a,j)
#                  print '^  < %d %d | dG | %d %d > :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( p,q,r,s,a,q,i,j,p,i,j,r,a,s,
#                                                               eta2B[idx2B[(a,q)],idx2B[(i,j)]],  W[idx3B[(p,i,j)],idx3B[(r,a,s)]]  )
#                  print '^                         :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( a,p,i,j,q,i,j,r,a,s,
#                                                               eta2B[idx2B[(a,p)],idx2B[(i,j)]],  W[idx3B[(q,i,j)],idx3B[(r,a,s)]]  )
#                  print '^                         :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( i,j,a,s,p,a,q,s,i,j,
#                                                               eta2B[idx2B[(i,j)],idx2B[(a,s)]] , W[idx3B[(p,a,q)],idx3B[(r,i,j)]]  )
#                  print '^                         :  eta(%d %d %d %d) W(%d %d %d %d %d %d) = %f  %f'%( i,j,a,r,p,a,q,r,i,j,
#                                                               eta2B[idx2B[(i,j)],idx2B[(a,r)]] , W[idx3B[(p,a,q)],idx3B[(s,i,j)]]  )
          dG3[bra,ket] += dG  
          dG3[ket,bra] = dG3[bra,ket]

  #SRS and [3,2]->2
  if np.linalg.norm(eta3B)>1e-5:
    for bra,(p,q) in enumerate( bas2B):
      for ket, (r,s) in enumerate(bas2B):
            if ket < bra : continue
            dG = 0.
            for i in holes:
              for a in particles:
                for b in particles:
                  dG += 0.5 * (-Gamma[idx2B[(i,q)],idx2B[(a,b)]] * eta3B[idx3B[(p,a,b)],idx3B[(r,i,s)]] 
                              + Gamma[idx2B[(i,p)],idx2B[(a,b)]] * eta3B[idx3B[(q,a,b)],idx3B[(r,i,s)]]
                              + Gamma[idx2B[(a,b)],idx2B[(i,s)]] * eta3B[idx3B[(p,i,q)],idx3B[(r,a,b)]]
                              - Gamma[idx2B[(a,b)],idx2B[(i,r)]] * eta3B[idx3B[(p,i,q)],idx3B[(s,a,b)]]  )
                for j in holes:
                   dG += 0.5 *(-Gamma[idx2B[(a,q)],idx2B[(i,j)]] * eta3B[idx3B[(p,i,j)],idx3B[(r,a,s)]]
                              + Gamma[idx2B[(a,p)],idx2B[(i,j)]] * eta3B[idx3B[(q,i,j)],idx3B[(r,a,s)]]
                              + Gamma[idx2B[(i,j)],idx2B[(a,s)]] * eta3B[idx3B[(p,a,q)],idx3B[(r,i,j)]]
                              - Gamma[idx2B[(i,j)],idx2B[(a,r)]] * eta3B[idx3B[(p,a,q)],idx3B[(s,i,j)]]  )
            dG3[bra,ket] += dG  
            dG3[ket,bra] = dG3[bra,ket]


  tstop = time.time()
#  print "elapsed: ",tstop-tstart

#  print "size of dG = ", np.linalg.norm(dG3)



  #SRS [1,3]->2 and [3,1]->2
#  print "[1,3]->2"
  tstart = time.time()
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          dG = 0.
          for i in holes:
            for a in particles:
              dG += ( eta1B[i,a] * W[idx3B[(a,p,q)],idx3B[(i,r,s)]] - f[i,a] * eta3B[idx3B[(a,p,q)],idx3B[(i,r,s)]]
                   -  eta1B[a,i] * W[idx3B[(i,p,q)],idx3B[(a,r,s)]] - f[a,i] * eta3B[idx3B[(i,p,q)],idx3B[(a,r,s)]]  )
#          dGamma[idx2B[(p,q)],idx2B[(r,s)]] += dG
          dG3[idx2B[(p,q)],idx2B[(r,s)]] += dG


  tstop = time.time()
#  print "elapsed: ",tstop-tstart

#  print "size of dG = ", np.linalg.norm(dG3)


  #SRS [3,3]->3
  #SRS This is the worst one. Very slow...
#  print "[3,3]->3"
  if np.linalg.norm(eta3B) > 1e-3:
    tstart = time.time()
    for bra, (p,q,r) in enumerate(bas3B):
      for ket, (s,t,u) in enumerate(bas3B):
        if ket < bra: continue
        pqr, stu = bra,ket
        dw = 0.
        for i in holes:
          for j in holes:
            for k in holes:
              ijk = idx3B[(i,j,k)]
              sij,tij,uij = idx3B[(s,i,j)], idx3B[(t,i,j)], idx3B[(u,i,j)]
              tku,sku,tks = idx3B[(t,k,u)], idx3B[(s,k,u)], idx3B[(t,k,s)]
              pqk,rqk,prk = idx3B[(p,q,k)], idx3B[(r,q,k)], idx3B[(p,r,k)]
              ijr,ijp,ijq = idx3B[(i,j,r)], idx3B[(i,j,p)], idx3B[(i,j,q)]
  
              dw += ( eta3B[pqr,ijk]  *     W[ijk, stu ]
                       -  W[pqr,ijk]  * eta3B[ijk, stu ] )
              dw += 0.5*( eta3B[pqk,sij]  *     W[ijr, tku]
                           -  W[pqk,sij]  * eta3B[ijr, tku]
                        - eta3B[rqk,sij]  *     W[ijp, tku]
                           +  W[rqk,sij]  * eta3B[ijp, tku]
                        - eta3B[prk,sij]  *     W[ijq, tku]
                           +  W[prk,sij]  * eta3B[ijq, tku]
                       -  eta3B[pqk,tij]  *     W[ijr, sku]
                           +  W[pqk,tij]  * eta3B[ijr, sku]
                        + eta3B[rqk,tij]  *     W[ijp, sku]
                           -  W[rqk,tij]  * eta3B[ijp, sku]
                        + eta3B[prk,tij]  *     W[ijq, sku]
                           -  W[prk,tij]  * eta3B[ijq, sku]
                       -  eta3B[pqk,uij]  *     W[ijr, tks]
                           +  W[pqk,uij]  * eta3B[ijr, tks]
                        + eta3B[rqk,uij]  *     W[ijp, tks]
                           -  W[rqk,uij]  * eta3B[ijp, tks]
                        + eta3B[prk,uij]  *     W[ijq, tks]
                           -  W[prk,uij]  * eta3B[ijq, tks]  )
        for a in particles:
          for b in particles:
            for c in particles:
              abc = idx3B[(a,b,c)]
              sab,tab,uab = idx3B[(s,a,b)], idx3B[(t,a,b)], idx3B[(u,a,b)]
              tcu,scu,tcs = idx3B[(t,c,u)], idx3B[(s,c,u)], idx3B[(t,c,s)]
              pqc,rqc,prc = idx3B[(p,q,c)], idx3B[(r,q,c)], idx3B[(p,r,c)]
              abr,abp,abq = idx3B[(a,b,r)], idx3B[(a,b,p)], idx3B[(a,b,q)]
  
              dw += ( eta3B[pqr,abc]  *     W[abc, stu ]
                       -  W[pqr,abc]  * eta3B[abc, stu ] )
              dw += 0.5*( eta3B[pqc,sab]  *     W[abr, tcu]
                           -  W[pqc,sab]  * eta3B[abr, tcu]
                        - eta3B[rqc,sab]  *     W[abp, tcu]
                           +  W[rqc,sab]  * eta3B[abp, tcu]
                        - eta3B[prc,sab]  *     W[abq, tcu]
                           +  W[prc,sab]  * eta3B[abq, tcu]
                       -  eta3B[pqc,tab]  *     W[abr, scu]
                           +  W[pqc,tab]  * eta3B[abr, scu]
                        + eta3B[rqc,tab]  *     W[abp, scu]
                           -  W[rqc,tab]  * eta3B[abp, scu]
                        + eta3B[prc,tab]  *     W[abq, scu]
                           -  W[prc,tab]  * eta3B[abq, scu]
                       -  eta3B[pqc,uab]  *     W[abr, tcs]
                           +  W[pqc,uab]  * eta3B[abr, tcs]
                        + eta3B[rqc,uab]  *     W[abp, tcs]
                           -  W[rqc,uab]  * eta3B[abp, tcs]
                        + eta3B[prc,uab]  *     W[abq, tcs]
                           -  W[prc,uab]  * eta3B[abq, tcs]  )
#        dW[pqr,stu] += dw
        dW[bra,ket] += dw
        dW[ket,bra] = dW[bra,ket]
                
    tstop = time.time()
#    print "elapsed: ",tstop-tstart

  #SRS [2,3]->3 and [3,2]->3
  #SRS Jesus, this is ugly...
#  print "[2,3]->3"
  if np.linalg.norm(eta3B) > 1e-3:
    tstart = time.time()
    for bra, (p,q,r) in enumerate(bas3B):
      for ket, (s,t,u) in enumerate(bas3B):
        if ket < bra: continue
        pqr,stu = bra,ket
        pq,pr,rq = idx2B[(p,q)], idx2B[(p,r)], idx2B[(r,q)]
        st,su,ut = idx2B[(s,t)], idx2B[(s,u)], idx2B[(u,t)]
        dw = 0.
        for a in particles:
          for b in particles:
            ab = idx2B[(a,b)]
            abp,abq,abr = idx3B[(a,b,p)], idx3B[(a,b,q)], idx3B[(a,b,r)]
            abS,abt,abu = idx3B[(a,b,s)], idx3B[(a,b,t)], idx3B[(a,b,u)]
            dw += 0.5 * ( eta2B[pq,ab] *     W[abr,stu]
                        -     W[pq,ab] * eta3B[abr,stu]
                        - eta2B[rq,ab] *     W[abp,stu]
                        +     W[rq,ab] * eta3B[abp,stu]
                        - eta2B[pr,ab] *     W[abq,stu]
                        +     W[pr,ab] * eta3B[abq,stu]
                        - eta2B[ab,st] *     W[pqr,abu]
                        -     W[ab,st] * eta3B[pqr,abu]
                        - eta2B[ab,ut] *     W[pqr,abS]
                        +     W[ab,ut] * eta3B[pqr,abS]
                        - eta2B[ab,su] *     W[pqr,abt]
                        +     W[ab,su] * eta3B[pqr,abt] )
        for a in holes:
          for b in holes:
            ab = idx2B[(a,b)]
            abS,abt,abu = idx3B[(a,b,s)], idx3B[(a,b,t)], idx3B[(a,b,u)]
            dw -= 0.5 * ( eta2B[pq,ab] *     W[abr,stu]
                        -     W[pq,ab] * eta3B[abr,stu]
                        - eta2B[rq,ab] *     W[abp,stu]
                        +     W[rq,ab] * eta3B[abp,stu]
                        - eta2B[pr,ab] *     W[abq,stu]
                        +     W[pr,ab] * eta3B[abq,stu]
                        - eta2B[ab,st] *     W[pqr,abu]
                        -     W[ab,st] * eta3B[pqr,abu]
                        - eta2B[ab,ut] *     W[pqr,abS]
                        +     W[ab,ut] * eta3B[pqr,abS]
                        - eta2B[ab,su] *     W[pqr,abt]
                        +     W[ab,su] * eta3B[pqr,abt] )

        for i in holes:
          pi,qi,ri = idx2B[(p,i)], idx2B[(q,i)], idx2B[(r,i)]
          its,itu,isu = idx3B[(i,t,s)], idx3B[(i,t,u)], idx3B[(i,s,u)]
          for a in particles:
            sa,ta,ua = idx2B[(s,a)], idx2B[(t,a)], idx2B[(u,a)]
            apr,aqr,aqp = idx3B[(a,p,r)], idx3B[(a,q,r)], idx3B[(a,q,p)]
            dw +=       ( eta2B[pi,sa] *     W[aqr,itu]
                        -     W[pi,sa] * eta3B[aqr,itu]
                        - eta2B[qi,sa] *     W[apr,itu]
                        +     W[qi,sa] * eta3B[apr,itu]
                        - eta2B[ri,sa] *     W[aqp,itu]
                        +     W[ri,sa] * eta3B[aqp,itu]
  
                        - eta2B[pi,ta] *     W[aqr,isu]
                        +     W[pi,ta] * eta3B[aqr,isu]
                        + eta2B[qi,ta] *     W[apr,isu]
                        -     W[qi,ta] * eta3B[apr,isu]
                        + eta2B[ri,ta] *     W[aqp,isu]
                        -     W[ri,ta] * eta3B[aqp,isu]
  
                        - eta2B[pi,ua] *     W[aqr,its]
                        +     W[pi,ua] * eta3B[aqr,its]
                        + eta2B[qi,ua] *     W[apr,its]
                        -     W[qi,ua] * eta3B[apr,its]
                        + eta2B[ri,ua] *     W[aqp,its]
                        -     W[ri,ua] * eta3B[aqp,its] )
        for a in holes:
          sa,ta,ua = idx2B[(s,a)], idx2B[(t,a)], idx2B[(u,a)]
          apr,aqr,aqp = idx3B[(a,p,r)], idx3B[(a,q,r)], idx3B[(a,q,p)]
          for i in particles:
            pi,qi,ri = idx2B[(p,i)], idx2B[(q,i)], idx2B[(r,i)]
            its,itu,isu = idx3B[(i,t,s)], idx3B[(i,t,u)], idx3B[(i,s,u)]
            dw +=       ( eta2B[pi,sa] *     W[aqr,itu]
                        -     W[pi,sa] * eta3B[aqr,itu]
                        - eta2B[qi,sa] *     W[apr,itu]
                        +     W[qi,sa] * eta3B[apr,itu]
                        - eta2B[ri,sa] *     W[aqp,itu]
                        +     W[ri,sa] * eta3B[aqp,itu]
  
                        - eta2B[pi,ta] *     W[aqr,isu]
                        +     W[pi,ta] * eta3B[aqr,isu]
                        + eta2B[qi,ta] *     W[apr,isu]
                        -     W[qi,ta] * eta3B[apr,isu]
                        + eta2B[ri,ta] *     W[aqp,isu]
                        -     W[ri,ta] * eta3B[aqp,isu]
  
                        - eta2B[pi,ua] *     W[aqr,its]
                        +     W[pi,ua] * eta3B[aqr,its]
                        + eta2B[qi,ua] *     W[apr,its]
                        -     W[qi,ua] * eta3B[apr,its]
                        + eta2B[ri,ua] *     W[aqp,its]
                        -     W[ri,ua] * eta3B[aqp,its] )
           
#        dW[pqr,stu] += dw
        dW[bra,ket] += dw
        dW[ket,bra] = dW[bra,ket]
                      
  
    tstop = time.time()
#    print "elapsed: ",tstop-tstart


  #SRS [1,3]->3 and [3,1]->3  for p in range(dim1B):
#  print "[1,3]->3"
  tstart = time.time()
  if np.linalg.norm(eta1B)>1e-5:
    for bra,(p,q,r) in enumerate(bas3B):
      for ket,(s,t,u) in enumerate(bas3B):
                if ket<bra: continue
                dw = 0.
                for a in range(dim1B):
                  dw += ( eta1B[p,a] * W[idx3B[(a,q,r)], ket] 
                        - eta1B[q,a] * W[idx3B[(a,p,r)], ket] 
                        - eta1B[r,a] * W[idx3B[(a,q,p)], ket] 
                        - eta1B[a,u] * W[bra, idx3B[(s,t,a)]] 
                        + eta1B[a,s] * W[bra, idx3B[(u,t,a)]] 
                        + eta1B[a,t] * W[bra, idx3B[(s,u,a)]]  )
  #              dW[pqr,stu] += dw
                dW[bra,ket] += dw
                dW[ket,bra] = dW[bra,ket]

  if np.linalg.norm(eta3B)>1e-5:
    for bra,(p,q,r) in enumerate(bas3B):
      for ket,(s,t,u) in enumerate(bas3B):
                if ket<bra: continue
                dw = 0.
                for a in range(dim1B):
                  dw -= ( f[p,a] * eta3B[idx3B[(a,q,r)], ket]
                        - f[q,a] * eta3B[idx3B[(a,p,r)], ket]
                        - f[r,a] * eta3B[idx3B[(a,q,p)], ket]
                        - f[a,u] * eta3B[bra, idx3B[(s,t,a)]]
                        + f[a,s] * eta3B[bra, idx3B[(u,t,a)]]
                        + f[a,t] * eta3B[bra, idx3B[(s,u,a)]] )
  #              dW[pqr,stu] += dw
                dW[bra,ket] += dw
                dW[ket,bra] = dW[bra,ket]


  tstop = time.time()
#  print "elapsed: ",tstop-tstart

  dGamma += dG3

#  bramax,ketmax = np.unravel_index( np.argmax(dW), dW.shape )
#  P,Q,R = bas3B[bramax]
#  S,T,U = bas3B[ketmax]
#  print 'max dW:', np.amax(dW), '    ', bramax,ketmax, '  =>   <%d %d %d || %d %d %d>'%(P,Q,R,S,T,U ) ,  '  check: ', dW[bramax,ketmax]
#  bramax,ketmax = np.unravel_index( np.argmax(W), W.shape )
#  P,Q,R = bas3B[bramax]
#  S,T,U = bas3B[ketmax]
#  print 'max W:', np.amax(W), '    ', bramax,ketmax, '  =>   <%d %d %d || %d %d %d>'%(P,Q,R,S,T,U ) ,  '  check: ', W[bramax,ketmax]
#  print '        < %d %d %d | W |%d %d %d > = %e'%(2,3,4,2,3,4, W[idx3B[(2,3,4)],idx3B[(2,3,4)]] )
#  print '        < %d %d %d | W |%d %d %d > = %e'%(3,4,2,3,4,2, W[idx3B[(3,4,2)],idx3B[(3,4,2)]] )
#  print '        < %d %d %d | W |%d %d %d > = %e'%(4,3,2,3,4,2, W[idx3B[(4,3,2)],idx3B[(3,4,2)]] )


  return dE, df, dGamma, dW

#-----------------------------------------------------------------------------------
# derivative wrapper
#-----------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, dim2B, dim3B):
  
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
  one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))

  ptr += dim1B*dim1B
  two_body = reshape(y[ptr:ptr+dim2B*dim2B], (dim2B, dim2B))

  ptr += dim2B*dim2B
  three_body = reshape(y[ptr:ptr+dim3B*dim3B], (dim3B, dim3B))

  return zero_body,one_body,two_body,three_body


def derivative_wrapper(t, y, user_data):

  dim1B = user_data["dim1B"]
  dim2B = dim1B*dim1B
  dim3B = dim1B*dim1B*dim1B


  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  basph2B   = user_data["basph2B"]
  idx2B     = user_data["idx2B"]
  idxph2B   = user_data["idxph2B"]
  idx3B     = user_data["idx3B"]
  occA_2B   = user_data["occA_2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]
  calc_eta  = user_data["calc_eta"]
  calc_rhs  = user_data["calc_rhs"]

  # extract operator pieces from solution vector
#  E, f, Gamma, W = get_operator_from_y(y, dim1B, dim2B)
  E, f, Gamma, W = get_operator_from_y(y, dim1B, dim2B, dim3B)


  # calculate the generator
#  eta1B, eta2B = calc_eta(f, Gamma, user_data)
  eta1B, eta2B, eta3B = calc_eta(f, Gamma, W, user_data)

  # calculate the right-hand side
#  dE, df, dGamma = calc_rhs(eta1B, eta2B, f, Gamma, user_data)
  dE, df, dGamma, dW = calc_rhs(eta1B, eta2B, eta3B, f, Gamma, W, user_data)

#  dW = np.zeros_like(W)

  # convert derivatives into linear array
#  dy   = np.append([dE], np.append(reshape(df, -1), reshape(dGamma, -1)))
  dy   = np.concatenate( ([dE], reshape(df,-1), reshape(dGamma,-1), reshape(dW,-1) ) )

  # share data
  user_data["dE"] = dE
  user_data["eta_norm"] = np.linalg.norm(eta1B,ord='fro')+np.linalg.norm(eta2B,ord='fro')
  
  return dy

#-----------------------------------------------------------------------------------
# pairing Hamiltonian
#-----------------------------------------------------------------------------------
def pairing_hamiltonian(delta, g, user_data):
  bas1B = user_data["bas1B"]
  bas2B = user_data["bas2B"]
  idx2B = user_data["idx2B"]

  dim = len(bas1B)
  H1B = np.zeros((dim,dim))

  for i in bas1B:
    H1B[i,i] = delta*np.floor_divide(i, 2)

  dim = len(bas2B)
  H2B = np.zeros((dim, dim))

  # spin up states have even indices, spin down the next odd index
  for (i, j) in bas2B:
    if (i % 2 == 0 and j == i+1):
      for (k, l) in bas2B:
        if (k % 2 == 0 and l == k+1):
          H2B[idx2B[(i,j)],idx2B[(k,l)]] = -0.5*g
          H2B[idx2B[(j,i)],idx2B[(k,l)]] = 0.5*g
          H2B[idx2B[(i,j)],idx2B[(l,k)]] = 0.5*g
          H2B[idx2B[(j,i)],idx2B[(l,k)]] = -0.5*g
  
  return H1B, H2B

#-----------------------------------------------------------------------------------
# normal-ordered pairing Hamiltonian
#-----------------------------------------------------------------------------------
def normal_order(H1B, H2B, user_data):
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # 0B part
  E = 0.0
  for i in holes:
    E += H1B[i,i]

  for i in holes:
    for j in holes:
      E += 0.5*H2B[idx2B[(i,j)],idx2B[(i,j)]]  

  # 1B part
  f = H1B
  for i in bas1B:
    for j in bas1B:
      for h in holes:
        f[i,j] += H2B[idx2B[(i,h)],idx2B[(j,h)]]  

  # 2B part
  Gamma = H2B

  return E, f, Gamma

#-----------------------------------------------------------------------------------
# Perturbation theory
#-----------------------------------------------------------------------------------
def calc_mbpt2(f, Gamma, user_data):
  DE2 = 0.0

  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  for i in holes:
    for j in holes:
      for a in particles:
        for b in particles:
          denom = f[i,i] + f[j,j] - f[a,a] - f[b,b]
          me    = Gamma[idx2B[(a,b)],idx2B[(i,j)]]
          DE2  += 0.25*me*me/denom

  return DE2

def calc_mbpt3(f, Gamma, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # DE3 = 0.0

  DE3pp = 0.0
  DE3hh = 0.0
  DE3ph = 0.0

  for a in particles:
    for b in particles:
      for c in particles:
        for d in particles:
          for i in holes:
            for j in holes:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[i,i] + f[j,j] - f[c,c] - f[d,d])
              me    = Gamma[idx2B[(i,j)],idx2B[(a,b)]]*Gamma[idx2B[(a,b)],idx2B[(c,d)]]*Gamma[idx2B[(c,d)],idx2B[(i,j)]]
              DE3pp += 0.125*me/denom

  for i in holes:
    for j in holes:
      for k in holes:
        for l in holes:
          for a in particles:
            for b in particles:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[l,l] - f[a,a] - f[b,b])
              me    = Gamma[idx2B[(a,b)],idx2B[(k,l)]]*Gamma[idx2B[(k,l)],idx2B[(i,j)]]*Gamma[idx2B[(i,j)],idx2B[(a,b)]]
              DE3hh += 0.125*me/denom

  for i in holes:
    for j in holes:
      for k in holes:
        for a in particles:
          for b in particles:
            for c in particles:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[j,j] - f[a,a] - f[c,c])
              me    = Gamma[idx2B[(i,j)],idx2B[(a,b)]]*Gamma[idx2B[(k,b)],idx2B[(i,c)]]*Gamma[idx2B[(a,c)],idx2B[(k,j)]]
              DE3ph -= me/denom
  return DE3pp+DE3hh+DE3ph

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

def main():
  # grab delta and g from the command line
  delta      = float(argv[1])
  g          = float(argv[2])

  particles  = 4

  # setup shared data
  dim1B     = 8

  # this defines the reference state
  # 1st state
  holes     = [0,1,2,3]
  particles = [ p for p in range(dim1B) if p not in holes ]
#  particles = [4,5,6,7]

  # 2nd state
  # holes     = [0,1,4,5]
  # particles = [2,3,6,7]

  # 3rd state
  # holes     = [0,1,6,7]
  # particles = [2,3,4,5]

  # basis definitions
  bas1B     = range(dim1B)
  bas2B     = construct_basis_2B(holes, particles)
  #basph2B   = construct_basis_ph2B(holes, particles)
  basph2B   = construct_basis_2B(holes, particles)

  bas3B     = construct_basis_3B(holes, particles)
  print "size of bas3B:", len(bas3B)

  idx2B     = construct_index_2B(bas2B)
  idxph2B   = construct_index_2B(basph2B)

  idx3B     = construct_index_3B(bas3B)

  # occupation number matrices
  occ1B     = construct_occupation_1B(bas1B, holes, particles)
  occA_2B   = construct_occupationA_2B(bas2B, occ1B)
  occB_2B   = construct_occupationB_2B(bas2B, occ1B)
  occC_2B   = construct_occupationC_2B(bas2B, occ1B)

  occphA_2B = construct_occupationA_2B(basph2B, occ1B)

  # store shared data in a dictionary, so we can avoid passing the basis
  # lookups etc. as separate parameters all the time
  user_data  = {
    "dim1B":      dim1B, 
    "holes":      holes,
    "particles":  particles,
    "bas1B":      bas1B,
    "bas2B":      bas2B,
    "basph2B":    basph2B,
    "bas3B":      bas3B,
    "idx2B":      idx2B,
    "idxph2B":    idxph2B,
    "idx3B":      idx3B,
    "occ1B":      occ1B,
    "occA_2B":    occA_2B,
    "occB_2B":    occB_2B,
    "occC_2B":    occC_2B,
    "occphA_2B":  occphA_2B,

    "eta_norm":   0.0,                # variables for sharing data between ODE solver
    "dE":         0.0,                # and main routine


#    "calc_eta":   eta_white,          # specify the generator (function object)
#    "calc_rhs":   flow_imsrg2         # specify the right-hand side and truncation
    "calc_eta":   eta_white_3B,       # specify the generator (function object)
    "calc_rhs":   flow_imsrg3         # specify the right-hand side and truncation
  }

  # set up initial Hamiltonian
  H1B, H2B = pairing_hamiltonian(delta, g, user_data)

  E, f, Gamma = normal_order(H1B, H2B, user_data) 

  #SRS initial 3b term is zero
  W = np.zeros((len(bas3B), len(bas3B)))


  # reshape Hamiltonian into a linear array (initial ODE vector)
#  y0   = np.append([E], np.append(reshape(f, -1), reshape(Gamma, -1)))
  y0   = np.concatenate( ([E], reshape(f,-1), reshape(Gamma,-1), reshape(W,-1)) )

  # integrate flow equations 
  solver = ode(derivative_wrapper,jac=None)
  solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
  solver.set_f_params(user_data)
  solver.set_initial_value(y0, 0.)

  sfinal = 50
  ds = 0.1

  print "%-8s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s"%(
    "s", "E" , "DE(2)", "DE(3)", "E+DE", "dE/ds", 
    "||eta||", "||fod||", "||Gammaod||", "||Wod||", "||W||")
  # print "-----------------------------------------------------------------------------------------------------------------"
  print "-" * (8+9*17)
  while solver.successful() and solver.t < sfinal:
    ys = solver.integrate(sfinal, step=True)
    
    dim2B = dim1B*dim1B
    dim3B = dim1B*dim1B*dim1B
#    E, f, Gamma = get_operator_from_y(ys, dim1B, dim2B)
    E, f, Gamma, W = get_operator_from_y(ys, dim1B, dim2B, dim3B)

    DE2 = calc_mbpt2(f, Gamma, user_data)
    DE3 = calc_mbpt3(f, Gamma, user_data)

    norm_fod     = calc_fod_norm(f, user_data)
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)
    norm_Wod     = calc_Wod_norm(W, user_data)
    norm_W       = np.linalg.norm(W)

    print "%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
      solver.t, E , DE2, DE3, E+DE2+DE3, user_data["dE"], user_data["eta_norm"], norm_fod, norm_Gammaod, norm_Wod, norm_W )
    if abs(DE2/E) < 10e-8: break


#    solver.integrate(solver.t + ds)

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()
