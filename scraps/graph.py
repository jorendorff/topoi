"""Directed graph algorithms.
"""

from collections import defaultdict

class Graph:
    """ Represents a directed graph.
    Nodes must be hashable.
    """

    def __init__(self, successors_fn=None):
        if successors_fn is not None:
            self.successors = successors_fn
        if not hasattr(self, "successors") or not callable(self.successors):
            raise ValueError("no successors method provided")

    def postorder_walk(graph, start_set):
        start_list = list(start_set)
        seen = set(start_list)
        result = []

        def walk(x):
            todo = []
            for y in self.successors(x):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
            for y in todo:
                walk(y)
            result.append(x)

        for x in start_list:
            walk(x)
        return result

    def compute_dominator_tree(self, start_nodes):
        """From <https://www.cs.rice.edu/~keith/EMBED/dom.pdf>."""

        start_set = set(start_nodes)
        nodes = list(reversed(self.postorder_walk(start_set)))
        order = {node: index for index, node in enumerate(nodes)}
        doms = {
            node: node if node in start_set else None
            for node in nodes
        }

        predecessors = defaultdict(list)
        for node in nodes:
            for succ in self.successors(node):
                predecessors[succ].append(node)

        def intersect(b1, b2):
            finger1 = b1
            finger2 = b2
            while finger1 != finger2:
                while order[finger1] < order[finger2]:
                    finger1 = doms[finger1]
                while order[finger2] < order[finger1]:
                    finger2 = doms[finger2]
            return finger1

        # compute the "first processed predecessor" of each node in a reverse
        # postorder walk
        first_processed_predecessor = {}
        for b in nodes:
            if b not in start_set:
                for c in predecessors[b]:
                    if c in first_processed_predecessor:
                        first_processed_predecessor[b] = c
                        break
                else:
                    assert False, "node has no visited predecessors despite walking in reverse postorder"

        start_node = self.blocks[self.start_id]
        start_doms[node] = start_node

        changed = True
        while changed:
            changed = False
            for b in nodes:
                if b not in start_set:
                    new_idom = first_processed_predecessor[b]
                    for p in predecessors[b]:
                        if p != new_idom:
                            if doms[p] is not None:
                                new_idom = intersect(p, new_idom)
                    if doms[b] != new_idom:
                        doms[b] = new_idom
                        changed = True

        return doms

    def recognize_ifs(self):
        for block in self.blocks:
            st_if = block.recognize_structured_if(self)
            if st_if is not None:
                ???
