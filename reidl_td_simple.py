#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:56:45 2019

@author: uddhav
"""

import networkx as nx 
import matplotlib.pyplot as plot 
from collections import defaultdict
import operator
import copy

def DFSTree(G, v=None):
    if v is None:
        nodes = G
    else:
        nodes = [v]
    visited=set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start,iter(G[start]))]
        while stack:
            parent,children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent,child
                    visited.add(child)
                    stack.append((child,iter(G[child])))
            except StopIteration:
                stack.pop()
    

def reidl_td_simple(G, t):
    infile = open(G)
    G = nx.Graph()

    for line in infile:
        edge = (line.split())
        if edge:
            if edge[0] == 'e':
                G.add_edge(int(edge[1]), int(edge[2]))
    
#    nx.draw(G, with_labels=True, font_weight='bold')
    
    Y = nx.Graph()
    
    root = next(DFSTree(G))[0]
    
    for i, j in DFSTree(G):
        Y.add_edge(i, j)
    
    spl = nx.shortest_path_length(Y, int(root))
    depth = max(spl.items(), key=operator.itemgetter(1))[1]
    
    if (depth > 2**t):
        return False
    
    #################return reidl-td()
    
    P = []
    subset = [root]
    pos = 0
    valList = list(spl.keys())
    
#    for key, value in spl.items():
#        pos += 1
#        if not subset and key not in subset and key not in P:
#            subset.append(key)
#        elif (spl[key] != spl[subset[-1]]):
#            subset.append(key)
#        elif (spl[key] == spl[subset[-1]]):
#            last = subset[-1]
#            subset.pop()
#            P.append(copy.deepcopy(subset))
#            subset.clear()
#            subset.append(P[-1][-1])
#            subset.append(last)
#            subset.append(key)
#            if (spl[key] != spl[valList[pos+1]]):
#                subset.clear()
    
    for i in range(1, len(spl)):
        if spl[valList[i]] != spl[valList[i-1]] and spl[valList[i]] != spl[valList[i+1]]:
            subset.append(valList[i])
        elif (spl[valList[i]] != spl[valList[i-1]] and spl[valList[i]] == spl[valList[i+1]]):
            P.append(copy.deepcopy(subset))
            subset.clear()
#            subset.append(P[-1][-1])
            subset.append(spl[valList[i]])
        elif (spl[valList[i]] == spl[valList[i-1]] and spl[valList[i]] != spl[valList[i+1]]):
            subset.append(spl[valList[i]])
            P.append(copy.deepcopy(subset))
            subset.clear()
        
    
    
    print(P)
    print(spl)