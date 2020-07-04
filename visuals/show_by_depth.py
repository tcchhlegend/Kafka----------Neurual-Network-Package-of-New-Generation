import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

class ShowGraph:
    __slots__ = 'nodes', 'for_flow'

    def show(self, with_labels=False):
        G = nx.DiGraph()
        pos = {}
        length = {}

        def add_node(node):
            print(node.name)
            print(G.nodes)
            if node in G.nodes:
                return False
            else:
                G.add_node(node)
                return True

        def add_pos(node, depth=0):
            if depth not in length.keys():
                length[depth] = 0
            if node not in pos.keys():
                pos[node] = (length[depth], depth)
                length[depth] += 1
            return depth + 1

        def add_edges(node):
            for c in node.children.values():
                G.add_edge(c, node)

        sizes = []

        def flow_control(node):
            status = add_node(node)
            if status:
                sizes.append(node.output_shape[0] * 100)

        self.for_flow(flow_control)
        self.for_flow(add_pos, fargs=0)
        self.for_flow(add_edges)

        print(len(sizes), type(sizes[0]), len(G.nodes))
        nx.draw(G, node_size=sizes, with_labels=with_labels, pos=pos)
        plt.show()

