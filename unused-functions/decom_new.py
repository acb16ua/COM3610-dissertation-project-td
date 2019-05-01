#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:30:22 2019

@author: uddhav
"""
import networkx as nx
import copy
from networkx.algorithms.approximation import treewidth
import itertools

infile = open('alarm.dgf')
G = nx.Graph()

for line in infile:
    edge = (line.split())
    if edge:
        if edge[0] == 'e':
            G.add_edge(int(edge[1]), int(edge[2]))

#nx.draw(G, with_labels=True, font_weight='bold')

#please, god = treewidth.treewidth_decomp(G)

#nx.draw(god, with_labels=False, font_weight='bold')
#print(god.nodes)


Y = nx.dfs_tree(G, 24)
T = nx.DiGraph.to_undirected(Y)
nx.draw(T, with_labels=True, font_weight='bold')
#
#please, god = treewidth.treewidth_decomp(T)
#
#nx.draw(god, with_labels=False, font_weight='bold')
#print(god.nodes)


def td(G):
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
    tw = len(first_bag) - 1
    
#    print(tree)
    
    
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

        # update treewidth
        tw = max(tw, len(new_bag) - 1)

        # add edge to decomposition (implicitly also adds the new node)
        decomp.add_edge(old_bag, new_bag)
        tree.append((nbrs))
    
          
#    print(decomp.nodes)
#    print(tree)
#    print(tree[1])
    return tree, decomp
        
#Y = nx.dfs_tree(G, 24)
#T = nx.DiGraph.to_undirected(Y)    
#root = 24
#    
#    
#spl = nx.shortest_path_length(Y, int(root))

#
#print(T[24])
#
#
#dont = treewidth.treewidth_decomp(T)
#print(dont)
#kid = treewidth.treewidth_min_degree(T)
#print(kid)
#me = treewidth.treewidth_min_fill_in(T)
#print(me)
#print(spl)

#valList = list(spl.values())
#keyList = list(spl.keys())
#tree = list()
#bag = list()
#
#for i in range(len(valList)-1):   
#    if (valList[i] != valList[i+1] and keyList[i] == root):
#        bag.append(keyList[i])
##        print(bag)
#    elif (valList[i] == valList[i+1]):
#        tree.append(copy.deepcopy(bag))
#        bag.clear()
##        print(i)
#        bag.append(tree[-1][-1])
#        bag.append(keyList[i])
#        bag.append(keyList[i+1])
#        i += 1
##        print(bag)
##        print(tree)
#    elif (valList[i] != valList[i+1]):
#        if keyList[i] in bag:
#            continue
##            tree.append(copy.deepcopy(bag))
##            bag.clear()
##            print(tree)
#        else:
#            bag.append(keyList[i])
##            print(bag)
#        
##print(tree)
#print(valList)   
#print(keyList)




#nodeList = list()    
#    
#nodesList = copy.deepcopy(T.nodes())
##print(nodesList)
#
#tree = []
#
#for i in nodesList:
##    print(i)
#    vertex = i
#    bag = list()
#    bag.append(vertex)
#    
#    for j in T.neighbors(i):
#        bag.append(j)
#    tree.append((bag))
#    T.remove_node(i)
#
#print(tree)           
    
poss, ible = td(G)


print(poss)
chonga = 0
for i in ible:
#    print(len(list(ible.neighbors(i))))
    if (len(list(ible.neighbors(i))) == 1):
        if (poss[chonga] == i):
            print(i)
            chonga += 1
    else:
        chonga += 1
        