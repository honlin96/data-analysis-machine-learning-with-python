# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:24:09 2024

@author: honlin
"""

import numpy as np
import numpy.linalg as LA

A = np.array([[1,2],[2,-2],[3,-1]])
B = np.array([[2,3],[4,5],[3,-5],[4,6]])

def cosine_similarity(A,B):
    '''
    A cosine similarity returns the pair-wise cosine 
    angle between the 2 vectors
    i.e. =  cos(theta) = v1.v2/|v1||v2|
    A: n x d array
    B: m x d array
    
    return: 
        cos: n x m array 
        
    '''

    no = A @ B.T 
    A_norm = np.array([LA.norm(A, axis = 1)])
    B_norm = np.array([LA.norm(B, axis = 1)])
    de = A_norm.T @ B_norm
    cos = np.divide(no, de)
    return cos

cos = cosine_similarity(A,B)

def euclidean_distance(A,B):
    '''
    euclidean_distance returns the pair-wise distance
    between 2 vectors
    i.e. C = A**2 - B**2

    Parameters
    ----------
    A : n x d array
    B : m x d array

    Returns
    -------
    dist: n x m array

    '''
    pairwise_sub = A[:, np.newaxis, :]-B[np.newaxis, :, :]
    dist = np.sqrt(np.sum(pairwise_sub**2, axis = 2))
    return dist

d = euclidean_distance(A,B)
