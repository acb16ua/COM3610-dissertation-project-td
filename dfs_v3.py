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
    
    
#    dfLabels = []
#    nextToVisit = []
#    parents = []
#    lastDfLabel = 0
#    stack = []
#    
#    for i in range(len(g.nodes)-1):
#        stack.append(i)
#        
#    while stack:
#        currentNode = stack[-1]
#        
#        if (dfLabels[currentNode] == 0):
#            lastDfLabel += 1
#            dfLabels[currentNode] = lastDfLabel
#            
#        if (nextToVisit[currentNode] == degree[currentNode]):
#            stack.pop
#        else:
            
    
    
    
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
        
for i in DFS('alarm.dgf'):
    print(i)

plt.subplot(122)
nx.draw(g_dfs, with_labels=True, font_weight='bold')

jumbo = DFS('alarm.dgf')
print(next(jumbo)[0])