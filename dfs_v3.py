# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:02:15 2019

@author: acb16ua
"""

import networkx as nx 
import matplotlib.pyplot as plt 
from collections import defaultdict


def DFS(g, v=None):
    
    infile = open(g)
    g = nx.Graph()

    for line in infile:
        edge = (line.split())
        if edge:
            if edge[0] == 'e':
                g.add_edge(int(edge[1]), int(edge[2]))
    
    plt.subplot(121) 
    nx.draw(g, with_labels=True, font_weight='bold')
    
    if v is None:
        nodes = g
    else:
        nodes = [v]
    visited=set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start,iter(g[start]))]
        while stack:
            parent,children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent,child
                    visited.add(child)
                    stack.append((child,iter(g[child])))
            except StopIteration:
                stack.pop()

 
g_dfs = nx.Graph()
                
for i, j in DFS('alarm.dgf'):
        g_dfs.add_edge(i, j)

plt.subplot(122)
nx.draw(g_dfs, with_labels=True, font_weight='bold')       