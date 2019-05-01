#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:07:09 2019

@author: uddhav
"""

import networkx as nx 
import math
import copy   
import random    
import glob
import signal
from networkx.algorithms.approximation import treewidth
import itertools
import time
import os
import operator

class TimeoutError(RuntimeError):
    pass

def handler(signum, frame):
    print("Timeout exception")
#    raise Exception("end of time!!")
    raise TimeoutError()

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
                
def tw(G):
    H = G.copy()
    
    maxmin = 0
    
    while (len(H.nodes) >= 2):
        v = min(H.degree, key=operator.itemgetter(1))[0]
        maxmin = max(maxmin, H.degree[v])
        H.remove_node(v)
        
    return maxmin    

        
def td(G):
    
    copG = G.copy()
    depth = 0

    if (len(copG.nodes) == 1):
        depth = 1
#        print('case1')
#        return depth
        
    elif (len(copG.nodes) > 1 and nx.is_connected(copG)):
        someList = []
#        print('case2')

        for v in list(copG.nodes)[:-1]:
            edges = list(copG.edges(v))
            copG.remove_node(v)            
            someList.append(td(copG))
            copG.add_node(v)
            copG.add_edges_from(edges)
                  
        depth = 1 + min(someList)
        
    elif(len(copG.nodes) > 1 and not nx.is_connected(copG)):
        someList2 = []
#        print('case3')
        for i in (copG.subgraph(c) for c in nx.connected_components(copG)):
            someList2.append(td(i))

        depth = max(someList2)
        
#    print(depth)
    
    return depth

def treeDecomposition(G):           
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
    tree.append(set(graph.keys()))
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
        tree.append((nbrs))
    
    return tree, decomp


#def td2(G):
#    
#    copG = G.copy()
#    depth = 0
#
#    if (len(copG.nodes) == 1):
#        depth = 1
#        print('case1')
##        return depth
#        
#    elif (len(copG.nodes) > 1):
#        minList = []
#        maxList = []
#        print('case2')
#
#        for v in list(copG.nodes):
#            edges = list(copG.edges(v))
#            copG.remove_node(v)
#            for i in (copG.subgraph(c) for c in nx.connected_components(copG)):
#                maxList.append(td2(i))            
#            minList.append(max(maxList))
#            copG.add_node(v)
#            copG.add_edges_from(edges)
#                  
#        depth = 1 + min(minList)
#        
#    print(depth)
#    return depth


#
#P = nx.Graph()
#P.add_nodes_from([1,2,3])
#P.add_edge(1,2)
#P.add_edge(1,3)
#
#Q = nx.path_graph(3)

#infile = open('alarm.dgf')
#G = nx.Graph()
#
#for line in infile:
#    edge = (line.split())
#    if edge:
#        if edge[0] == 'e':
#            G.add_edge(int(edge[1]), int(edge[2]))


def original(): 
    samples = []
    
    path = '/Users/uddhav/University/NewStuff/ThirdYear/COM3610/graphs'
    filenames = glob.glob(path + '/*.dgf')
    resultFile = open('results_libtw_tw.txt', 'a+')
    for filename in filenames:
        cpu_time = time.time()
    #    print(filename)
        
        infile = open(filename)
        
        G = nx.Graph()
        
        result = 0
        
        for line in infile:
            edge = (line.split())
            if edge:
                if edge[0] == 'e':
                    G.add_edge((edge[1]), (edge[2]))
                    
        samples.append(G)
        
        lb = tw(G)
        
        dfs = nx.Graph()
        for i, j in DFSTree(G):
            dfs.add_edge(i, j)
            
        for node in dfs.nodes:
            spl = nx.shortest_path_length(dfs, node)
            break
        
        ub = max(spl.items(), key=operator.itemgetter(1))[1]
#        lb = math.log(ub+2,2)
        
    #    signal.signal(signal.SIGALRM, handler)
    #    
    #    signal.alarm(5)
        
        instance = os.path.basename(filename)
        instance = instance.split('.')
        instance = instance[0]
        
        
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(1)
            result = td(G)
        except TimeoutError as ex:
            print(':(')
         
        cpu_time = time.time() - cpu_time    
        resultFile.write('\n\n-----------------------------------------')
        resultFile.write('\nCONDITIONS: \nGraph file = %s \n|V| = %i \n|E| = %i \nAlgorithm = 2^n \n' % (instance, len(G.nodes),len(G.edges)))
        if result != 0:
            resultFile.write('\nTree-depth = %i\n' %(result))
            resultFile.write('CPU Time = %f\n' %(cpu_time))
        else:
            resultFile.write('\nTree-depth = TIMEOUT EXCEPTION\n')
            resultFile.write('CPU Time = %f\n' %(cpu_time))
            resultFile.write('Lowerbound = %i\n' %(lb))
            resultFile.write('Upperbound = %i\n' %(ub))
        
    resultFile.close()
 


#infile = open('Moser.edge')
#        
#G = nx.Graph()
#
#result = 0
#
#for line in infile:
#    edge = (line.split())
#    if edge:
#        if edge[0] == 'e':
#            G.add_edge((edge[1]), (edge[2]))
#          
#G_Dfs = nx.dfs_tree(G)
#G_Dfs = nx.DiGraph.to_undirected(G_Dfs)
#diameter = nx.eccentricity(G_Dfs)
#td_upper = max(list(diameter.values()))
    
    
    
    
    
#Se = list()
#X = set()
#e = 0.2
#S_size = math.ceil((0.5 - e)*len(G.nodes))
#
#
#
#for i in G.nodes:  
#    for j in G.neighbors(i):
#        X.add(i)
#        if len(X) <= S_size:
#            X.add(j)
#            
#    Se.append(copy.deepcopy(X))
#    X.clear()
#
#print(Se)
#
#subGraphs = list()
#
#for i in Se:
#    subGraphs.append(G.subgraph(list(i)))
#
#T = nx.minimum_spanning_tree(G)
#
##seTd = []
##for i in subGraphs:
##    seTd.append(td(i))
#
###### PROBLEMATIC VERTEX #####
#root = random.randint(1,len(T.nodes))
##root = list(T.nodes)[0]
#
#for i in T.nodes:
#    if i == root:
#        continue
#    else:
#        copyT = T.copy()
#        sp = nx.shortest_path(copyT, source=root, target=i)
#        print("source = %i /n target = %i", (root, i))
#        print(sp)
#        copyT.remove_nodes_from(sp[:-1])
#        print(len(copyT)>S_size)
#        print(len(sp[:-1]))
#        
#        if (len(copyT)>S_size and len(sp[:-1])>S_size) :
#            problemV = i
#            print(problemV)
#            
##            