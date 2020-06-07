class DAGNode:
    """计算图节点
    有向无环图
    没有children的节点为叶节点

    ATTRIBUTE
    parents : 字典，存储节点的父节点
    children : 字典，存储结点的子节点

    METHOD
    is_leaf(self) : 判断是否为叶子结点
    is_root(self) : 判断是否为根结点
    for_flow(self, func) : 前向递归流动，从若干起始点向父节点递归调用func函数
    back_flow(self, func) : 反向递归流动，向子节点递归调用func函数
    """
    def __init__(self):
        self.parents = {}
        self.children = {}

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parents == {}

    def for_flow(self, func, fargs=None):
        if fargs is not None:
            fargs = func(self, fargs)
        else:
            func(self)
        for p in self.parents.values():
            p.for_flow(func, fargs=fargs)

    def back_flow(self, func, fargs=None):
        if fargs is not None:
            fargs = func(self, fargs)
        else:
            func(self)
        for c in self.children.values():
            c.back_flow(func, fargs=fargs)


class DAGraph:
    """计算图
    有向无环图

    ATTRIBUTE
    nodes : 图中的所有节点
    leaves : 图中的叶子节点
    roots : 图中的根节点

    METHOD
    _check(nodes: list) -> list | None : 只有当列表中的所有元素都是DAGNode的实例时，才返回该列表，否则报错
    get_leaves(self) : 检查并返回所有叶子节点（用于初始化和更新）
    get_roots(self) : 检查并返回所有根节点（用于初始化和更新）
    for_flow(self, func, starts=None) : 前向递归流动，若传入starts则以starts为起始点，否则以leaves为起始点
    back_flow(self, func, ends=None) : 反向递归流动，同上
    get_subgraph(self, starts : list, ends : list, mode='forward') : 返回连接 starts 和 ends 的子图
    """
    def __init__(self, nodes):
        self.nodes = self._check(nodes)
        self.leaves = self.get_leaves()
        self.roots = self.get_roots()

    def _check(self, nodes):
        if all(isinstance(node, DAGNode) for node in nodes):
            return nodes
        else:
            raise ValueError("All nodes should be instances of class `DAGNode`.")

    def get_leaves(self):
        return [node for node in self.nodes if node.is_leaf()]

    def get_roots(self):
        return [node for node in self.nodes if node.is_root()]

    def for_flow(self, func, starts=None, fargs=None):
        if not starts:
            for l in self.leaves:
                l.for_flow(func, fargs=fargs)
        else:
            starts = self._check(starts)
            for s in starts:
                s.for_flow(func, fargs=fargs)

    def back_flow(self, func, ends=None, fargs=None):
        if not ends:
            for r in self.roots:
                r.back_flow(func, fargs=fargs)
        else:
            ends = self._check(ends)
            for e in ends:
                e.back_flow(func, fargs=fargs)

    def get_subgraph(self, starts : list, ends : list, mode='forward'):
        nodes = starts + ends
        def func(node):
            if node in ends:
                node.tmp_state =  True
            elif node.is_root():
                node.tmp_state = False
            else:
                if mode == 'forward':
                    flow_to = node.parents.values()
                else:
                    flow_to = node.children.values()
                node.tmp_state = any([func(p) for p in flow_to])
                if node.tmp_state and (node not in nodes):
                    nodes.append(node)
            return node.tmp_state
        if mode == 'forward':
            self.for_flow(func, starts=starts)
        elif mode == 'backward':
            self.back_flow(func, ends=ends)
        else:
            raise NotImplementedError('type `%s` in get_subgraph is not implemented' % type)

        return nodes