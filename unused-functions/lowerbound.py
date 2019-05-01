#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:12:16 2019

@author: uddhav
"""

import networkx as nx
import operator
from networkx.algorithms.approximation import treewidth

infile = open('alarm.dgf')
G = nx.Graph()

for line in infile:
    edge = (line.split())
    if edge:
        if edge[0] == 'e':
            G.add_edge(int(edge[1]), int(edge[2]))
            

def tw(G):
    H = G.copy()
    
    maxmin = 0
    
    while (len(H.nodes) >= 2):
        v = min(H.degree, key=operator.itemgetter(1))[0]
        maxmin = max(maxmin, H.degree[v])
        H.remove_node(v)
        
    print(maxmin)
    
