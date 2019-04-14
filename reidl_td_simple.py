#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:56:45 2019

@author: uddhav
"""

import networkx as nx 
import matplotlib.pyplot as plot 
from collections import defaultdict

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
    
    nx.draw(G, with_labels=True, font_weight='bold')
    
    GDfs = nx.Graph()
    
    for i, j in DFSTree(G):
        GDfs.add_edge(i, j)