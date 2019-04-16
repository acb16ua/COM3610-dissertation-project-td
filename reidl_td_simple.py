#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:56:45 2019

@author: uddhav
"""

import networkx as nx 
import matplotlib.pyplot as plot 
from collections import defaultdict
from networkx.algorithms.approximation import treewidth
import operator
import copy
import itertools

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

def treeDecomposition(G):
#    T = nx.Graph()
#    nodeList = copy.deepcopy(G.nodes())    
#    tree = list()
#    
#    for i in nodeList:
#        vertex = i
#        bag = list()
#        bag.append(vertex)
#    
#        for j in T.neighbors(i):
#            bag.append(j)
#        tree.append((bag))
#        T.add_node(frozenset(bag))
#        G.remove_node(i)
#        
#    for n1 in range(len(nodeList)):
#        checker = False
#        for n2 in range(n1+1, len(nodeList)):
#            if checker == False and len(nodeList[n1].intersection(nodeList[n2])) > 0:
#                T.add_edge(nodeList[n1], nodeList[n2])
#                checker = True
#    
#    nx.draw(T, with_labels=True, font_weight='bold')
           
    tree = []
    
    graph = {n: set(G[n]) - set([n]) for n in G}
    
    node_stack = []
    
    elim_node = treewidth.min_fill_in_heuristic(graph)
    
    while elim_node is not None:
        neigh = graph[elim_node]
        for u, v in itertools.permutations(neigh, 2):
            if v not in graph[u]:
                graph[u].add(v)
                
        node_stack.append((elim_node, neigh))
        
        for u in graph[elim_node]:
            graph[u].remove(elim_node)
            
        del graph[elim_node]
        elim_node = treewidth.min_fill_in_heuristic(graph)
        
    decomp = nx.Graph()
    first_bag = frozenset(graph.keys())
    tree.append(graph.keys())
    decomp.add_node(first_bag)    
    
    while node_stack:
        # get node and its neighbors from the stack
        (curr_node, nbrs) = node_stack.pop()

        # find a bag all neighbors are in
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break

        if old_bag is None:
            # no old_bag was found: just connect to the first_bag
            old_bag = first_bag

        # create new node for decomposition
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)


        # add edge to decomposition (implicitly also adds the new node)
        decomp.add_edge(old_bag, new_bag)
        tree.append(list(nbrs))
    
    return tree, decomp
        


def reidl_td(G_p, t):
    
    T_p = treeDecomposition(G_p)
    X = T_p[0]
    r = 'u'
    
    G_p.add_node(r)
    
    for n in G_p:
        if (n != r):
            G_p.add_edge(n, r)
            
    G = G_p
    
    T = T_p
    
    for i in range(len(T)):
        T[i].append(r)
        
    
    R = reidl_td_rec(G, T, t)
    
    return R is not None
    

def reidl_td_rec(G, T, t):
    R = set()
    
    for X in len(T):
        if (len(X) == 1):
            r = X[0]
            F = [r]
            h = 1
            R.add((F, {r}, h))
        elif isForget(X):
            R.add()
        elif isIntroduce(X):
            R.add()    
        elif isJoin(X):    
            R.add()
    
    return R


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