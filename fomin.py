#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:07:09 2019

@author: uddhav
"""

import networkx as nx 
import math
import copy

def tdAux(S):
    depth = 0
    
    if len(S) == 1:
        depth = 1
    else:
        S.pop()
        depth = 1 + tdAux(copy.deepcopy(S))
    
    return depth 
        
        
def td(G):
    
    copG = G.copy()
    depth = 0

    if (len(copG.nodes) == 1):
        depth = 1
        print('case1')
#        return depth
        
    elif (len(copG.nodes) > 1 and nx.is_connected(copG)):
        someList = []
        print('case2')

        for v in list(copG.nodes)[:-1]:
            edges = list(copG.edges(v))
            copG.remove_node(v)            
            someList.append(td(copG))
#            print(copG.nodes)
            copG.add_node(v)
#            print(copG.nodes)
            copG.add_edges_from(edges)
                  
        depth = 1 + min(someList)
        
    elif(len(copG.nodes) > 1 and not nx.is_connected(copG)):
        someList2 = []
        print('case3')
        for i in (copG.subgraph(c) for c in nx.connected_components(copG)):
            someList2.append(td(i))

        depth = max(someList2)
        
    print(depth)
    return depth



P = nx.Graph()
P.add_nodes_from([1,2,3])
P.add_edge(1,2)
P.add_edge(1,3)

Q = nx.path_graph(3)

infile = open('alarm.dgf')
G = nx.Graph()

for line in infile:
    edge = (line.split())
    if edge:
        if edge[0] == 'e':
            G.add_edge(int(edge[1]), int(edge[2]))
            

Se = list()
X = set()
e = 0.2
S_size = math.floor((0.5 - e)*len(G.nodes))


#for i in range(1, S_size):
#    for j in G.nodes:
#        
#        if (len(S) <= i):
#            S.add(j)
#        else:
#            Se.append(copy.deepcopy(S))
#            S.clear()
#            S.add(j)


#for i in range(1, len(G.nodes)):
#    for j in G.nodes:
#        
#        if (i == 1):
#            X.add(j)
#            Se.append(copy.deepcopy(X))
#            X.clear()
#        elif():
#            Se.append(copy.deepcopy(S))
#            S.clear()
#            S.add(j)


for i in G.nodes:  
    for j in G.neighbors(i):
        X.add(i)
        if len(X) <= S_size:
            X.add(j)
            
    Se.append(copy.deepcopy(X))
    X.clear()

print(Se)

tdSet = list()

for i in Se:
    tdSet.append(tdAux(copy.deepcopy(i)))

#for i in range(math.floor((0.25 + (1.5*e))*len(G.nodes))):
#    for j in Se


T = nx.minimum_spanning_tree(G)

#### FINDING PROBLEMATIC VERTEX#####

