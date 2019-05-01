# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:46:10 2018

@author: acb16ua
"""

import networkx as nx 
import matplotlib.pyplot as plot

def dfsSearch(graph, startNode, endNode):

    infile = open(graph)
    g = nx.Graph()

    for line in infile:
        edge = (line.split())
        if edge:
            if edge[0] == 'e':
                g.add_edge(int(edge[1]), int(edge[2]))
                
#    stack = [(startNode, [strtNode])]
#    while stack:



    visited = set()
    nodes = [startNode]
    
#    print(visited)
#    for start in nodes:
#        if start in visited:
#            continue
#        visited.add(start)
##        print(visited)
#        stack = [(start, endNode, iter(g[start]))]
##        print(stack)
#        while stack:
#            parent, depth_now, children = stack[-1]
#            try:
#                child = next(children)
#                if child not in visited:
#                    yield parent, child
#                    visited.add(child)
#                    if depth_now > 1:
#                        stack.append((child, depth_now - 1, iter(g[child])))
#            except StopIteration:
#                stack.pop()
    
    
    
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
#        print(visited)
        stack = [(start, endNode, iter(g[start]))]
#        print(stack)
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child 
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(g[child])))

            except StopIteration:
                stack.pop()



                     

    print(g.number_of_nodes())
    print(g.number_of_edges())
    print(list(g.nodes))  #lists the nodes
    print(list(g.edges))  #lists the edges
#    print(list(g.adj[34])) #lists the neighbours
#    print(g.degree[34])    #returns the degree
    print(g[2])
#    print(g[2][14])
                
#    plot.subplot(g)
#    plot.show()
    nx.draw(g, with_labels=True, font_weight='bold')
            
#    stack, path = [strtNode], []
#    
#    while stack:
#    
#    stack = [(strtNode, [strtNode])]
#    while stack:
#        (vertex, path) = stack.pop()
#        for next in graph[vertex] - set(path):
#            if next == endNode:
#                yield path + [next]
#            else:
#                stack.append((next, path + [next]))

        
for i in dfsSearch('alarm.dgf', 6, 7):
    print(i)
