#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 00:11:56 2019

@author: uddhav
"""

import networkx as nx
import signal
from networkx.algorithms.approximation import treewidth
import itertools
import time
import os
import operator
import argparse
import math

class TimeoutError(RuntimeError):
    pass

def handler(signum, frame):
    print("Timeout exception")
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
        
    elif (len(copG.nodes) > 1 and nx.is_connected(copG)):
        someList = []

        for v in list(copG.nodes)[:-1]:
            edges = list(copG.edges(v))
            copG.remove_node(v)            
            someList.append(td(copG))
            copG.add_node(v)
            copG.add_edges_from(edges)
                  
        depth = 1 + min(someList)
        
    elif(len(copG.nodes) > 1 and not nx.is_connected(copG)):
        someList2 = []
        for i in (copG.subgraph(c) for c in nx.connected_components(copG)):
            someList2.append(td(i))

        depth = max(someList2)
    
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


def parse_args():
    parser = argparse.ArgumentParser(description='%(prog)s -f instance')
    parser.add_argument('-f', '--file', dest='instance', action='store', type=lambda x: os.path.realpath(x),
                        default=None, help='instance')
    parser.add_argument('-o', '--timeout', dest='timeout', action='store', type=int, default=500,
                        help='timeout for each SAT call')
    args = parser.parse_args()
    return args


def main(): 
    args = parse_args()
    samples = []
    instance = args.instance
    instance = os.path.realpath(instance)
    resultFile = open('results_naive.txt', 'a+')
    cpu_time = time.time()
    timeout = args.timeout
    #    print(instance)
    
    infile = open(instance)
    
    G = nx.Graph()
    
    result = 0
    
    for line in infile:
        edge = (line.split())
        if edge:
            if edge[0] == 'e':
                G.add_edge((edge[1]), (edge[2]))
                
    samples.append(G)
    
#    lb = tw(G)
    
    dfs = nx.Graph()
    for i, j in DFSTree(G):
        dfs.add_edge(i, j)
        
    for node in dfs.nodes:
        spl = nx.shortest_path_length(dfs, node)
        break
    
    ub = max(spl.items(), key=operator.itemgetter(1))[1]
    lb = math.log(ub+2,2)
    
    
    instance = os.path.basename(instance)
    instance = instance.split('.')
    instance = instance[0]
    
    
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        result = td(G)
    except TimeoutError as ex:
        print(':(')
     
    cpu_time = time.time() - cpu_time    
    resultFile.write('\n\n-----------------------------------------')
    resultFile.write('\nCONDITIONS: \nGraph file = %s \n|V| = %i \n|E| = %i \nAlgorithm = Naive DP Algorithm \n' % (instance, len(G.nodes),len(G.edges)))
    if result != 0:
        resultFile.write('\nTree-depth = %i\n' %(result))
        resultFile.write('CPU Time = %f\n' %(cpu_time))
        resultFile.write('Lowerbound = %i\n' %(lb))
        resultFile.write('Upperbound = %i\n' %(ub))
    else:
        resultFile.write('\nTree-depth = TIMEOUT EXCEPTION\n')
        resultFile.write('CPU Time = %f\n' %(cpu_time))
        resultFile.write('Lowerbound = %i\n' %(lb))
        resultFile.write('Upperbound = %i\n' %(ub))
    
    resultFile.close()

if __name__ == "__main__":
    main()            