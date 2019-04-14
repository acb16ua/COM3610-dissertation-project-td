# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:49:51 2019

@author: acb16ua
"""

import networkx as nx 
#from networkx.algorithms import approximation
#from networkx.algorithms.approximation import clique
from networkx.algorithms.approximation import treewidth
import matplotlib.pyplot as plt 

infile = open('alarm.dgf')
g = nx.Graph()

for line in infile:
    edge = (line.split())
    if edge:
        if edge[0] == 'e':
            g.add_edge(int(edge[1]), int(edge[2]))

plt.subplot(121)    
nx.draw(g, with_labels=True, font_weight='bold')

#bing = clique.max_clique(g)
tw, decomp_graph = treewidth.treewidth_min_fill_in(g)

print(tw)
#print(bing)
plt.subplot(122)
nx.draw(decomp_graph, with_labels=False, font_weight='bold')
