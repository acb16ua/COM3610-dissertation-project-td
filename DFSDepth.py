#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:07:01 2019

@author: uddhav
"""
import networkx as nx
import matplotlib.pyplot as plt
 
class DFSDepth(object):
    def __init__(self, parents):
        self._parents = parents
        self._n = len(parents)
        self._max_depth = None
        self._depths = [None] * self._n

    def max_depth(self):
        if self._max_depth is not None:
            return self._max_depth
        for idx, parent in enumerate(self._parents):
            parent_stack = []
            while parent != -1 and self._depths[idx] is None:
                parent_stack.append(idx)
                idx, parent = parent, self._parents[parent]
            if parent == -1:
                depth = 1
            else:
                depth = self._depths[idx]
            while parent_stack:
              self._depths[parent_stack.pop()] = depth
              depth += 1
            if self._max_depth < depth:
                self._max_depth = depth
        return self._max_depth

    def get_depth(self, idx):
        depth = self._depths[idx]
        if depth is not None:
            return depth
        parent = self._parents[idx]
        if parent == -1:
            depth = 1
        else:
            depth = self.get_depth(parent) + 1
        self._depths[idx] = depth
        return depth
    
infile = open('alarm.dgf')
g = nx.Graph()

for line in infile:
    edge = (line.split())
    if edge:
        if edge[0] == 'e':
            g.add_edge(int(edge[1]), int(edge[2]))
    
plt.subplot(121) 
nx.draw(g, with_labels=True, font_weight='bold')

print(DFSDepth(g_dfs).max_depth())    