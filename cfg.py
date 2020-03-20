import basic
from basic import Program
import argparse
import re


def indent_stmts(stmts, indent='    '):
    return re.sub(r'(?m)^(?!$)', indent, stmts)


assert indent_stmts('') == ''
assert indent_stmts('a\nb') == '    a\n    b'
assert indent_stmts('a\nb\n') == '    a\n    b\n'


class FallThroughGotoStmt(basic.GotoStmt):
    """Implied GOTO statement at the end of a block, so that control falls through
    to the next block in program order.
    """
    def stmt_code(self):
        return "REM fall through to line {}".format(self.target)


class StructuredIfStmt(basic.Stmt):
    """IF condition THEN stmts ELSE stmts END IF"""
    def __init__(self, condition, then_stmts, else_stmts):
        super(StructuredIfStmt, self).__init__()
        self.condition = condition
        self.then_stmts = then_stmts
        self.else_stmts = else_stmts

    def __str__(self):
        if self.lineno is not None:
            lineno = '%05d   ' % self.lineno
        else:
            lineno = ' ' * 8
        s = '{}IF {} THEN\n'.format(lineno, self.condition)
        s += indent_stmts(''.join(str(stmt) + '\n'
                                  for stmt in self.then_stmts))
        if self.else_stmts:
            s += '        ELSE\n'
            s += indent_stmts(''.join(str(stmt) + '\n'
                                      for stmt in self.else_stmts))
        s += '        END IF'
        return s

    def jump_targets(self):
        for block in (self.then_stmts, self.else_stmts):
            for stmt in block:
                for target in stmt.jump_targets():
                    yield target


OP_OPPOSITES = {
    '=': '<>',
    '<>': '=',
    '<': '>=',
    '>=': '<',
    '>': '<=',
    '<=': '>',
}


def negate_boolean_expr(expr):
    if isinstance(expr, basic.ComparisonExpr):
        op = OP_OPPOSITES[expr.op]
        return basic.ComparisonExpr(op, expr.left, expr.right)
    raise ValueError("don't know how to negate expression: " + str(expr))


class BasicBlock:
    """Get it?"""

    def __init__(self, body_stmts, branch_stmts, successors):
        self.id = (body_stmts or branch_stmts)[0].lineno
        self.body_stmts = body_stmts
        self.branch_stmts = branch_stmts
        self.successors = successors
        self.predecessors = []
        self.order = None
        self.doms = None

    def dump(self, cfg):
        if self.doms is not None:
            print("'(dominator: {})".format(self.doms.id))
        if self.predecessors:
            print("'(from: {})".format(', '.join(map(str, self.predecessors))))
        elif self.id != cfg.start_id:
            print("'(unreachable)")
        for line in self.body_stmts:
            print(line)
        for line in self.branch_stmts:
            print(line)
        if self.successors:
            def is_back_edge(cid):
                return cfg.blocks[cid].dominates(self)
            edges = ["{}{}".format("*" if is_back_edge(cid) else "", cid)
                     for cid in self.successors]
            print("'(to: {})".format(', '.join(edges)))
        print()

    def dominates(self, other):
        node = other
        while node is not self and node is not None and node.doms is not node:
            node = node.doms
        return node is self

    def under_what_condition(self, successor_id):
        pre = None
        for stmt in self.branch_stmts:
            if isinstance(stmt, basic.IfStmt):
                if stmt.target == successor_id:
                    if pre is not None:
                        raise ValueError("can't combine multiple conditions")
                    return stmt.condition
                else:
                    pre = negate_boolean_expr(stmt.condition)
            elif isinstance(stmt, basic.GotoStmt):
                if pre is None:
                    raise ValueError("unconditional branch; shouldn't happen")
                return pre
            elif isinstance(stmt, basic.OnGotoStmt):
                if successor_id not in stmt.targets:
                    raise ValueError("successor {} not reachable from line:\n{}"
                                     .format(successor_id, stmt))
                if pre is not None:
                    raise ValueError("can't combine mutiple conditions")
                v = stmt.targets.index(successor_id) + 1
                return basic.ComparisonExpr('=', stmt.expr, basic.NumberLiteralExpr(v))
            else:
                break
        raise ValueError(
            "can't figure out under what condition these lines branch "
            "to line {}:\n{}"
            .format(successor_id, ''.join(str(stmt) for stmt in self.branch_stmts)))

    def recognize_structured_if(self, cfg):
        if len(self.successors) == 2:
            b_id, c_id = self.successors
            b = cfg.blocks[b_id]
            c = cfg.blocks[c_id]
            if b_id in c.successors:
                b, b_id, c, c_id = c, c_id, b, b_id
            if (b.successors == [c_id]
                and b.predecessors == [self.id]
                and set(c.predecessors) == {self.id, b.id}):
                return self, b, c
        return None

    def match(self, pattern):
        """Test whether this node and its neighbors match a given pattern.

        The pattern is a string like "AB AC BC *A C*". Each letter stands for
        a distinct block, such that:
        -   `AB` means A has successor B.
        -   `AC` means A has successor C.
        -   `BC` means B has successor C.

        There is an implied constraint that all matched Blocks have exactly the
        predecessors or successors listed in the pattern, and no others; rules
        containing stars relax this constraint:
        -   `*A` means A is an "entry block"; it may have any number
            of predecessors that aren't A or B.
        -   `C*` means C is an "exit block"; it may have any number
            of successors that aren't B or C.
        However, since we have both `*A` and `C*`, it *is* allowed for C to
        have successor A.

                    A   B   C   other
                A   no  yes yes  no
                B   no  no  yes  no
                C   ok  no  no   ok
            other   ok  no  no

        If possible, this method returns a dictionary mapping 'A' to self
        and each other letter in the pattern to a Block such that the
        constraints are satisfied. Otherwise this returns None.

        """
        symbols = set()
        entries = set()  # nodes with *A rules
        exits = set()    # nodes with A* rules
        for word in pattern.split():
            [a, b] = word
            if a == '*':
                entries.add(b)
            elif a.isupper():
                symbols.add(a)
            else:
                raise ValueError("invalid character " + repr(a)
                                 + " in pattern: " + repr(pattern))
            if b == '*':
                exits.add(a)
            elif b.isupper():
                symbols.add(b)
            else:
                raise ValueError("invalid character " + repr(b)
                                 + " in pattern: " + repr(pattern))
        if 'A' not in symbols:
            raise ValueError("pattern must contain A: " + repr(pattern))
        raise NotImplementedError("sorry lol")


class Cfg:
    """Control flow graph."""

    def __init__(self, blocks, start_id):
        self.blocks = blocks
        self.start_id = start_id

        # Populate `.predecessors` for each block.
        for b in self.blocks.values():
            for id in b.successors:
                self.blocks[id].predecessors.append(b.id)

        # Populate `.order` for each block.
        for i, node in enumerate(self._postorder()):
            node.order = i

        self._compute_dominator_tree()

    def dump(self):
        for block in self.blocks.values():
            block.dump(self)

    def _postorder(self):
        seen = set()

        def walk(id):
            if id in seen:
                return
            seen.add(id)
            b = self.blocks[id]
            for c in b.successors:
                yield from walk(c)
            yield b

        return list(walk(self.start_id))

    def _reverse_postorder(self):
        return reversed(self._postorder())

    def _compute_dominator_tree(self):
        """From <https://www.cs.rice.edu/~keith/EMBED/dom.pdf>."""

        def intersect(b1, b2):
            finger1 = b1
            finger2 = b2
            while finger1 != finger2:
                while finger1.order < finger2.order:
                    finger1 = finger1.doms
                while finger2.order < finger1.order:
                    finger2 = finger2.doms
            return finger1

        # compute the "first processed predecessor" of each node in a reverse
        # postorder walk
        first_processed_predecessor = {
            self.start_id: None
        }

        for b in self._reverse_postorder():
            if b.id != self.start_id:
                for c_id in b.predecessors:
                    if c_id in first_processed_predecessor:
                        first_processed_predecessor[b.id] = self.blocks[c_id]
                        break
                else:
                    assert False, ("node has no visited predecessors despite "
                                   "walking in reverse postorder")

        for b in self.blocks.values():
            b.doms = None
        start_node = self.blocks[self.start_id]
        start_node.doms = start_node

        changed = True
        while changed:
            changed = False
            for b in self._reverse_postorder():
                if b is not start_node:
                    new_idom = first_processed_predecessor[b.id]
                    for p_id in b.predecessors:
                        p = self.blocks[p_id]
                        if p is not new_idom:
                            if p.doms is not None:
                                new_idom = intersect(p, new_idom)
                    if b.doms is not new_idom:
                        b.doms = new_idom
                        changed = True

    def fix_structured_if(self, structured_if):
        if_block, then_block, join_block = structured_if
        new_stmt = StructuredIfStmt(
            if_block.under_what_condition(then_block.id),
            then_block.body_stmts,
            [])

        # Bunch of tricky mutation to merge the three blocks.
        if_block.body_stmts.append(new_stmt)
        if_block.body_stmts += join_block.body_stmts
        if_block.branch_stmts = join_block.branch_stmts
        if_block.successors = join_block.successors
        for successor_id in if_block.successors:
            b = self.blocks[successor_id]
            index = b.predecessors.index(join_block.id)
            b.predecessors[index] = if_block.id
        del self.blocks[then_block.id]
        del self.blocks[join_block.id]

    def structure_ifs(self):
        changed = True
        while changed:
            changed = False
            for block in self.blocks.values():
                structured_if = block.recognize_structured_if(self)
                if structured_if is not None:
                    self.fix_structured_if(structured_if)
                    changed = True
                    break  # we mutated self.blocks while iterating over it

        # Merging blocks messes up the dominator tree. Regenerate it.
        self._compute_dominator_tree()


def program_to_cfg(program):
    # TODO - take a graph-walking approach to computing jump_targets, to ignore
    # unreachable jumps
    #
    # TODO - unify each EmptyStmt with the next statement, even if there are
    # jump targets
    #
    # TODO - join A and B if A.successors == [B] and B.predecessors == [A].

    if len(program.lines) == 0:
        raise ValueError("program is empty, nothing to do")

    jump_targets = program.all_jump_targets()

    blocks = {}
    current_body = []
    current_branches = []

    def compute_successors(stmts, next_lineno):
        successors = []
        if len(current_branches) == 0:
            successors.append(next_lineno)
        else:
            last = current_branches[-1]
            # GOSUB counts as falling through. This is maybe a little shabby,
            # but it's just right for the dominator graph.
            if (next_lineno is not None
                and (last.can_fall_through()
                     or isinstance(last, basic.GosubStmt))):
                current_branches.append(FallThroughGotoStmt(next_lineno))
                successors.append(next_lineno)
            if isinstance(last, basic.GotoStmt) and len(current_branches) > 1:
                prev = current_branches[-2]
                if isinstance(prev, (basic.IfStmt, basic.NextStmt)):
                    successors += list(prev.jump_targets())
            successors += list(last.jump_targets())
        return successors

    def cut(next_lineno):
        nonlocal current_body, current_branches
        assert len(current_body) + len(current_branches) > 0
        for stmt in (current_body + current_branches)[1:]:
            stmt.lineno = None
        block = BasicBlock(current_body,
                           current_branches,
                           compute_successors(current_branches, next_lineno))
        assert block.id not in blocks
        blocks[block.id] = block
        current_body = []
        current_branches = []

    prev = None
    for stmt in program.lines:
        if len(current_body) + len(current_branches) > 0:
            if stmt.lineno in jump_targets or isinstance(prev, basic.ForStmt):
                # Cut before any jump target, for sure.
                need_cut = True
            elif isinstance(prev, (basic.IfStmt, basic.NextStmt)):
                # Cut after any IF or NEXT statement, unless followed by GOTO.
                need_cut = not isinstance(stmt, basic.GotoStmt)
            elif current_branches:
                # Cut after anything else that jumps or ends the program.
                need_cut = True
            else:
                need_cut = False
            if need_cut:
                cut(stmt.lineno)
        if ((len(stmt.jump_targets()) > 0 or not stmt.can_fall_through())
            and not isinstance(stmt, basic.GosubStmt)):
            current_branches.append(stmt)
        else:
            current_body.append(stmt)
        prev = stmt
    cut(None)
    return Cfg(blocks, start_id=program.lines[0].lineno)


def main():
    parser = argparse.ArgumentParser(
        description="Try to recover structure from an unstructured BASIC "
                    "program.")
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help="filename of the BASIC program to analyze")
    options = parser.parse_args()
    [filename] = options.file
    program = Program.load(filename)
    cfg = program_to_cfg(program)
    cfg.structure_ifs()
    cfg.dump()


if __name__ == '__main__':
    main()
