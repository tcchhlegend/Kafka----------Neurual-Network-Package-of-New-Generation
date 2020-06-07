import networkx as nx
from matplotlib import pyplot as plt

class ShowGraph:
    __slots__ = 'nodes', 'for_flow'

    def show(self, layerfold=True):
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        pos = {}
        length = {}
        def add_pos(node, depth=0):
            if depth not in length.keys():
                length[depth] = 0
            if node not in pos.keys():
                pos[node] = (length[depth], depth)
                length[depth] += 1
            return depth + 1
        self.for_flow(add_pos, fargs=0)

        def add_edges(node):
            for c in node.children.values():
                G.add_edge(c, node)
        self.for_flow(add_edges)
        nx.draw(G, with_labels=True, pos=pos)
        plt.show()

