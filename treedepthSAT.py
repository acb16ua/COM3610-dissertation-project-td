#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:36:26 2019

@author: uddhav
"""
import networkx as nx
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import subprocess
from satispy.solver import Minisat
import time
import pycosat
from pathlib import Path
#from fomin import *

def apex_vertex(g):
    buff = 0
    delete_vertices = list()
    for u, degree in g.degree():
        if degree == g.number_of_nodes() - 1:
            delete_vertices.append(u)
    g.remove_nodes_from(delete_vertices)
    buff += len(delete_vertices)
    nx.convert_node_labels_to_integers(g, first_label=0)
    return g, buff

def degree_one_reduction(g):
    nodes = set()
    for u in g.nodes():
        deg = 0
        for v in g.neighbors(u):
            if g.degree(v) == 1:
                if deg == 0:
                    deg = 1
                else:
                    nodes = nodes.union({v})
    g.remove_nodes_from(list(nodes))
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    return g

def variables(g,width):
    nv=g.number_of_nodes()
    p=[[[0 for i in range(width)]for j in range(nv)] for k in range(nv)]
    nvar=1
    for u in range(nv):
        for v in range(u,nv):
            for i in range(width):
                p[u][v][i]=nvar
                nvar+=1
    return p,nvar-1


def generate_encoding(g,width):
    s,nvar=variables(g,width)
    encoding=''
    nclauses=0
    nv=g.number_of_nodes()
    for u in range(nv):
        for v in range(u,nv):
            encoding += '%i 0\n' % (s[u][v][width-1])
            nclauses += 1
            encoding += '-%i 0\n' % (s[u][v][0])
            nclauses += 1
    for u in range(nv):
        for v in range(u, nv):
            for i in range(1,width):
                encoding += '-%i %i 0\n' % (s[u][v][i-1], s[u][v][i])
                nclauses += 1
    for u in range(nv):
        for v in range(u + 1, nv):
            for w in range(v + 1, nv):
                for i in range(width):
                    encoding += '-%i -%i %i 0\n' % (s[u][v][i], s[u][w][i], s[v][w][i])
                    nclauses += 1
                    encoding += '-%i -%i %i 0\n' % (s[u][v][i], s[v][w][i], s[u][w][i])
                    nclauses += 1
                    encoding += '-%i -%i %i 0\n' % (s[u][w][i], s[v][w][i], s[u][v][i])
                    nclauses += 1
    for u in range(nv):
        for v in range(u + 1, nv):
            for i in range(width):
                    encoding += '-%i %i 0\n' % (s[u][v][i], s[u][u][i])
                    encoding += '-%i %i 0\n' % (s[u][v][i], s[v][v][i])
                    nclauses += 2
    for u in range(nv):
        for v in range(u+1,nv):
            for i in range(1,width):
                encoding+='-%i %i %i 0\n'%(s[u][v][i],s[u][u][i-1],s[v][v][i-1])
                nclauses+=1
    for e in g.edges():
        u=min(e)
        v=max(e)
        for i in range(1,width):
            encoding+='-%i %i -%i %i 0\n'%(s[u][u][i],s[u][u][i-1],s[v][v][i],s[u][v][i])
            nclauses+=1
            encoding+='-%i %i -%i %i 0\n'%(s[u][u][i],s[v][v][i-1],s[v][v][i],s[u][v][i])
            nclauses+=1
    preamble='p cnf %i %i\n'%(nvar,nclauses)
    return preamble+encoding

def decode_output(sol,g,width):
    with open(sol, 'r') as out_file:
        out = out_file.read()
    out = out.split('\n')
    out = out[1]
    out = out.split(' ')
    out = list(map(int, out))
    out.pop()
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    s,nvar = variables(g, width)
    components=list()
    for i in range(width-1,0,-1):
        level=list()
        for u in range(nv):
            ver=list()
            for v in range(u,nv):
                if out[s[u][v][i]-1]>0:
                    ver.append(v)
            do_not_add=0
            for v in level:
                if set(ver).issubset(set(v)):
                    do_not_add=1
            if do_not_add==0:
                level.append(ver)
        components.append(level)
    for i in components:
        sys.stderr.write(str(i)+'\n')
    sys.stderr.write('\n'+"*"*10+'\n')
    decomp=nx.DiGraph()
    root=list()
    level_i=list()
    for i in range(width-1,0,-1):
        level = list()
        # sys.stderr.write('\n'+'*'*10+'\n')
        for u in range(nv):
            if out[s[u][u][i]-1]>0 and out[s[u][u][i-1]-1]<0:
                edge_add=False
                if i==width-1:
                    root.append(u)
                decomp.add_node(u,level=i)
                # sys.stderr.write("%i "%u)
                level.append(u)
                if level_i!=[]:
                    for v in level_i[len(level_i)-1]:
                        if out[s[min(u,v)][max(u,v)][i+1]-1]>0:
                            decomp.add_edge(v,u)
                            edge_add=True
                if not edge_add:
                    if level_i!=[]:
                        level_u=g.number_of_nodes()
                        v_u=-1
                        for v in decomp.nodes(data=True):
                            if v[0]==u:
                                continue
                            if out[s[min(u,v[0])][max(u,v[0])][v[1]['level']]-1]>0:
                                if level_u>v[1]['level']:
                                    level_u=v[1]['level']
                                    v_u=v[0]
                        decomp.add_edge(v_u,u)
        level_i.append(level)
        # print level
#    show_graph(decomp,1)
    verify_decomp(g=g,s=decomp,width=width,root=root)


def verify_decomp(g, s, width, root):
    sys.stderr.write("\nValidating tree depth decomposition\n")
    sys.stderr.flush()
    # print g.edges()
    for e in g.edges():
        try:
            nx.shortest_path(s, e[0], e[1])
        except:
            try:
                nx.shortest_path(s, e[1], e[0])
            except:
                raise Exception("Edge %i %i not covered\n" % (e[0], e[1]))
    for v, d in g.degree():
        count = 0
        if d != 1:
            continue
        for i in root:
            try:
                if len(nx.shortest_path(s, i, v)) - 1 > width:
                    raise ValueError("depth of tree more than width\n")
            except:
                count += 1
                if count == len(root):
                    raise Exception("No root found for %i\n"%v)
                continue
    sys.stderr.write("Valid treedepth decomp\n")
    sys.stderr.flush()


def show_graph(graph, layout, nolabel=0):
    """ show graph
    layout 1:graphviz,
    2:circular,
    3:spring,
    4:spectral,
    5: random,
    6: shell
    """

    m = graph.copy()
    if layout == 9991:
        pos = nx.nx_agraph.graphviz_layout(m)
    elif layout == 2:
        pos = nx.circular_layout(m)
    elif layout == 3:
        pos = nx.spring_layout(m)
    elif layout == 4:
        pos = nx.spectral_layout(m)
    elif layout == 5:
        pos = nx.random_layout(m)
    elif layout == 6:
        pos = nx.shell_layout(m)
    if not nolabel:
        nx.draw_networkx_edge_labels(m, pos)
    nx.draw_networkx_labels(m, pos)
    nx.draw_networkx_nodes(m, pos)
    # write_dot(m, "m1.dot")
    # os.system("dot -Tps m1.dot -o m1.ps")
#    nx.draw(m, pos)
    plt.show()
    
    
def main():
    cpu_time = time.time()
#    instance = 'famous/Holt.edge/'
    instance = 'graphs/celar09pp.dgf/'
#    instance = 'alarm.dgf'
#    instance = 'alarm.dgf'
    instance = os.path.realpath(instance)
    d = -1
    solver = "/usr/local/Cellar/minisat/2.2.0_2/bin/minisat"
    width = -1
    temp = '/Users/uddhav/University/NewStuff/ThirdYear/COM3610/temp/'
    infile = open(instance)
    g = nx.Graph()

    for line in infile:
        edge = (line.split())
        if edge:
            if edge[0] == 'e':
                g.add_edge((edge[1]), (edge[2]))
                
    instance = os.path.basename(instance)
    instance = instance.split('.')
    instance = instance[0]
    
    timeout = 500
    n = g.number_of_nodes()
    m = g.number_of_edges()
    prep_time = time.time()
    g=degree_one_reduction(g=g)
    buff = 0
    lb = 0
    ub = 0
    to = False
    g,buff=apex_vertex(g=g)

    print('treedepthp2sat', instance, n, m,g.number_of_nodes(),buff)
    if g.number_of_nodes() <= 1:
        print(lb,ub,to)
        exit(0)
    encoding_time = list()
    solving_time = list()
    g=nx.relabel.convert_node_labels_to_integers(g,first_label=0)
    if width == -1:
        for i in range(g.number_of_nodes()+2,1,-1):
            encode_time = time.time()
            encoding = generate_encoding(g, i)
            cnf = temp + instance + '_' + str(i) + ".cnf"
            with open(cnf, 'w') as ofile:
                ofile.write(encoding)
            # with open(cnf,'r') as ifile:
            #     s=ifile.read()
            #     print s
            encode_time = time.time() - encode_time
            encoding_time.append(encode_time)
            sol = temp + instance + '_' + str(i) + '.sol'
#            Path(temp + instance + '_' + str(i) + '.txt').touch()
#            anotherSol = temp + instance + '_' + str(i) + '.txt'
            cmd = [solver, '-cpu-lim=%i' % timeout, cnf, sol]
            # print cmd
            solving = time.time()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = p.communicate()
            rc = p.returncode
            solving = time.time() - solving
            solving_time.append(solving)
            sys.stderr.write('*' * 10+'\n')
#            print(output)
#            print(err)
            sys.stderr.write("\n%i %i\n"%(i-1, rc))
            if rc == 0:
                to = True
                if lb == 0:
                    ub = i
            if rc == 10:
                if to:
                    ub = i
                    lb = i-2
                decode_output(sol=sol, g=g, width=i)
                print(i-2, lb, ub, to, time.time() - cpu_time, prep_time, sum(
                    encoding_time), sum(solving_time), end=' ')
                for j in solving_time:
                    print(j, end=' ')
#            print(i-2, lb, ub, to)

#                exit(0)

if __name__ == "__main__":
    main()