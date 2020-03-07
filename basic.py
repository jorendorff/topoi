# basic.py - Run BASIC programs.

import re, random, sys, math
import argparse

TOKEN_RE = re.compile(r'''(?x)
\s*
(REM[ ].* | '.* | [A-Z][0-9A-Z]*\$? | [0-9]+ | "[^"]*" | <> | .)
\s*
''')


def print_line(lineno, line):
    print("%05d   %s" % (lineno, line))


class BasicError(ValueError):
    pass


def basic_to_str(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, float):
        # Print "1", not "1.0".
        if int(value) == value:
            value = int(value)

        # My best guess is that positive numbers were printed with a space in
        # front.
        if value > 0:
            return ' ' + str(value)
        return str(value)
    else:
        raise TypeError("printing unrecognized value: " + repr(value))


assert basic_to_str(1.0) == " 1"


# Expressions

class NumberLiteralExpr:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        if self.value % 1.0 == 0.0:
            return str(int(self.value))
        return str(self.value)

    def evaluate(self, env):
        return self.value

    def type_check(self):
        return 'number'


class StringLiteralExpr:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '"' + self.value + '"'

    def evaluate(self, env):
        return self.value

    def type_check(self):
        return 'string'


class RndExpr:
    def __init__(self):
        pass

    def __str__(self):
        return 'RND'

    def evaluate(self, env):
        return random.random()

    def type_check(self):
        return 'number'


class IdentifierExpr:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def type_check(self):
        return 'string' if self.name.endswith('$') else 'number'

    def evaluate(self, env):
        name = self.name
        if name not in env.variables:
            env.variables[name] = "" if name.endswith('$') else 0.0
        return env.variables[name]

    def assign(self, env, value):
        name = self.name
        expected_type = str if name.endswith("$") else float
        assert isinstance(value, expected_type)
        if env.tracing:
            print("*** Setting {} to {}".format(name, basic_to_str(value)))
        env.variables[name] = value


BASIC_FUNCTIONS = {
    'INT': (['number'], 'number'),
    'LEN': (['string'], 'number'),
    'INSTR': (['number', 'string', 'string'], 'number'),
    'MID$': (['string', 'number', 'number'], 'string')
}


class CallExpr:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __str__(self):
        return "{}({})".format(self.name, ",".join(str(arg) for arg in self.args))

    def type_check(self):
        actual_arg_types = [arg.type_check() for arg in self.args]
        if self.name in BASIC_FUNCTIONS:
            expected_arg_types, rtype = BASIC_FUNCTIONS[self.name]
            if actual_arg_types != expected_arg_types:
                raise BasicError(
                    "{} expects {} argument(s): {}; got {}"
                    .format(self.name, len(expected_arg_types), ", ".join(expected_arg_types),
                            ", ".join(actual_arg_types)))
            return rtype
        elif len(self.name.rstrip('$')) <= 2:
            if actual_arg_types != ['number']:
                raise BasicError("array argument must be number")
            return 'string' if self.name.endswith('$') else 'number'
        else:
            raise BasicError("unrecognized function: " + self.name)

    def evaluate(self, env):
        args = [e.evaluate(env) for e in self.args]
        if self.name == 'INT':
            [n] = args
            return float(math.floor(n))
        elif self.name == 'LEN':
            [s] = args
            return float(len(s))
        elif self.name == 'INSTR':
            start, haystack, needle = args
            if start < 1 or int(start) != start:
                raise BasicError("INSTR expects string index, not {}".format(start))
            where = haystack.find(needle, int(start) - 1) + 1
            return float(where)
        elif self.name == 'MID$':
            s, start, length = args
            if int(start) != start or start < 1 or int(length) != length or length < 0:
                raise BasicError("invalid bounds in MID$")
            start = int(start) - 1
            length = int(length)
            return s[start:start + length]
        elif self.name in env.arrays or len(self.name.rstrip('$')) <= 2:
            if not isinstance(args[0], float):
                raise BasicError("array expects numeric index")
            if self.name not in env.arrays:
                env.define_array(self.name, 10)
            i = int(args[0])
            arr = env.arrays[self.name]
            if not (0 <= i < len(arr)):
                raise BasicError("array index out of range: {}({})".format(self.name, i))
            return arr[i]
        else:
            raise BasicError("unknown function: " + self.name)

    def assign(self, env, value):
        args = [e.evaluate(env) for e in self.args]
        if self.name in BASIC_FUNCTIONS:
            raise BasicError("can't assign to {}()".format(self.name))
        elif self.name in env.arrays or len(self.name) <= 3:
            i = int(args[0])
            env.set_elem(self.name, i, value)
        else:
            raise BasicError("unknown array: " + self.name)


class ArithmeticExpr:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        return "({}{}{})".format(self.left, self.op, self.right)

    def type_check(self):
        if self.left.type_check() != 'number' or self.right.type_check() != 'number':
            raise BasicError("{} expects numeric argument".format(self.op))
        return 'number'

    def evaluate(self, env):
        left = self.left.evaluate(env)
        if not isinstance(left, float):
            raise BasicError("{} expects numeric argument".format(self.op))
        right = self.right.evaluate(env)
        if not isinstance(right, float):
            raise BasicError("{} expects numeric argument".format(self.op))
        if self.op == '+':
            return left + right
        elif self.op == '-':
            return left - right
        elif self.op == '*':
            return left * right
        elif self.op == '/':
            return left / right
        else:
            raise BasicError("unknown operator: " + self.op)


class ComparisonExpr:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        return "({}{}{})".format(self.left, self.op, self.right)

    def type_check(self):
        left_t = self.left.type_check()
        right_t = self.right.type_check()
        if left_t != right_t:
            raise BasicError(
                "{} expects arguments of the same type, but {} is {} and {} is {}"
                .format(self.op, self.left, left_t, self.right, right_t))
        return 'number'

    def evaluate(self, env):
        left = self.left.evaluate(env)
        right = self.right.evaluate(env)
        if self.op == '<':
            cond = left < right
        elif self.op == '>':
            cond = left > right
        elif self.op == '=':
            cond = left == right
        elif self.op == '<>':
            cond = left != right
        else:
            raise BasicError("unknown operator: " + self.op)
        return -1.0 if cond else 0.0


# Statements

class Stmt:
    def jump_targets(self):
        return ()

    def check_line_numbers(self, lineno_table):
        for lineno in self.jump_targets():
            if lineno not in lineno_table:
                raise BasicError("no such line number: " + repr(lineno))

    def type_check(self):
        pass


class EmptyStmt(Stmt):
    def __init__(self):
        pass

    def __str__(self):
        return "REM"

    def run(self, env):
        pass


class DimStmt(Stmt):
    def __init__(self, name, size_expr):
        self.name = name
        self.size_expr = size_expr

    def __str__(self):
        return "DIM {}({})".format(self.name, self.size_expr)

    def run(self, env):
        size = self.size_expr.evaluate(env)
        if type(size) is not float:
            raise BasicError("array dimension must be numeric")
        elif size < 0 or float(int(size)) != size:
            raise BasicError("invalid array dimension")
        else:
            env.define_array(self.name, int(size))


class EndStmt(Stmt):
    def __init__(self):
        pass

    def __str__(self):
        return "END"

    def run(self, env):
        env.status = 'stop'


class StopStmt(Stmt):
    def __init__(self):
        pass

    def __str__(self):
        return "STOP"

    def run(self, env):
        env.status = 'stop'


class RandomizeStmt(Stmt):
    def __init__(self):
        pass

    def __str__(self):
        return "RANDOMIZE"

    def run(self, env):
        pass


class IfStmt(Stmt):
    def __init__(self, condition, target):
        self.condition = condition
        self.target = target

    def __str__(self):
        return "IF {} THEN {}".format(self.condition, self.target)

    def type_check(self):
        if self.condition.type_check() != 'number':
            raise BasicError("IF statement condition must be number")

    def jump_targets(self):
        return [self.target]

    def run(self, env):
        cond = self.condition.evaluate(env)
        if type(cond) is not float:
            raise BasicError("IF requires a numeric condition")
        if cond != 0.0:
            env.jump(self.target)


class ForStmt(Stmt):
    def __init__(self, var, first_expr, last_expr):
        self.var = var
        self.first_expr = first_expr
        self.last_expr = last_expr

    def __str__(self):
        return "FOR {} = {} TO {}".format(self.var, self.first_expr, self.last_expr)

    def type_check(self):
        if self.var.endswith('$'):
            raise BasicError("FOR loop variable must be a number variable")
        if self.first_expr.type_check() != 'number' or self.last_expr.type_check() != 'number':
            raise BasicError("FOR loop bounds must be numbers")

    def run(self, env):
        first_value = self.first_expr.evaluate(env)
        if type(first_value) is not float:
            raise BasicError("FOR requires numeric bounds")
        env.variables[self.var] = first_value
        last_value = self.last_expr.evaluate(env)
        if type(last_value) is not float:
            raise BasicError("FOR requires numeric bounds")
        env.stack.append(['FOR', self.var, first_value, last_value, env.get_next_index()])


class NextStmt(Stmt):
    def __init__(self, var=None):
        self.var = var

    def __str__(self):
        if self.var is None:
            return "NEXT"
        else:
            return "NEXT " + self.var

    def type_check(self):
        if self.var.endswith('$'):
            raise BasicError("NEXT variable must be a number variable")

    def run(self, env):
        if len(env.stack) == 0:
            raise BasicError("NEXT without FOR")
        record = env.stack[-1]
        if record[0] != 'FOR':
            raise BasicError("NEXT without FOR")
        if self.var is not None and self.var != record[1]:
            raise BasicError("FOR/NEXT variable mismatch ({}/{})".format(record[1], self.var))
        value = record[2] + 1.0
        if value <= record[3]:
            env.variables[self.var] = record[2] = value
            env.jump_to_index(record[4])
        else:
            env.stack.pop()


class PrintTab:
    def __str__(self):
        return ","

    def type_check(self):
        return 'string'


class PrintStmt(Stmt):
    def __init__(self, exprs, leading_comma_count=0, trailing_semicolon=False):
        self.exprs = exprs
        self.leading_comma_count = leading_comma_count
        self.trailing_semicolon = trailing_semicolon

    def __str__(self):
        if self.exprs == [] and self.leading_comma_count == 0:
            return "PRINT"
        return "PRINT{}{}".format(
            "," * self.leading_comma_count or " ",
            ''.join(str(e) for e in self.exprs))

    def type_check(self):
        for expr in self.exprs:
            expr.type_check()

    def run(self, env):
        # maybe these are tabs, worth a shot
        sys.stdout.write("\t" * self.leading_comma_count)
        for e in self.exprs:
            if isinstance(e, PrintTab):
                sys.stdout.write("\t")
            else:
                sys.stdout.write(basic_to_str(e.evaluate(env)))
        if not self.trailing_semicolon:
            sys.stdout.write("\n")


class LinputStmt(Stmt):
    def __init__(self, var):
        self.var = var

    def __str__(self):
        return "LINPUT {}".format(self.var)

    def type_check(self):
        if not self.var.endswith('$'):
            raise BasicError("LINPUT variable must be a string variable")

    def run(self, env):
        line = input()
        env.variables[self.var] = line


class OnGotoStmt(Stmt):
    def __init__(self, expr, targets):
        self.expr = expr
        self.targets = targets

    def __str__(self):
        return "ON {} GOTO {}".format(self.expr, ','.join(str(t) for t in self.targets))

    def jump_targets(self):
        return self.targets

    def type_check(self):
        if self.expr.type_check() != 'number':
            raise BasicError("ON/GOTO requires numeric operand")

    def run(self, env):
        num = self.expr.evaluate(env)
        if num < 0 or int(num) != num:
            raise BasicError("invalid ON/GOTO operand")
        index = int(num)
        if not (0 < index <= len(self.targets)):
            raise BasicError("ON/GOTO operand out of range: " + str(index))
        env.jump(self.targets[index - 1])


class GotoStmt(Stmt):
    def __init__(self, target):
        self.target = target

    def __str__(self):
        return "GOTO {}".format(self.target)

    def jump_targets(self):
        return [self.target]

    def run(self, env):
        env.jump(self.target)


class GosubStmt(Stmt):
    def __init__(self, target):
        self.target = target

    def __str__(self):
        return "GOSUB {}".format(self.target)

    def jump_targets(self):
        return [self.target]

    def run(self, env):
        env.stack.append(['GOSUB', env.get_next_index()])
        env.jump(self.target)


class ReturnStmt(Stmt):
    def __init__(self):
        pass

    def __str__(self):
        return "RETURN"

    def run(self, env):
        if len(env.stack) == 0:
            raise BasicError("RETURN without GOSUB")
        record = env.stack.pop()
        if record[0] != 'GOSUB':
            raise BasicError("RETURN without GOSUB")
        env.jump_to_index(record[1])


class AssignmentStmt(Stmt):
    def __init__(self, targets, expr):
        self.targets = targets
        self.expr = expr

    def __str__(self):
        return "{}={}".format("=".join(str(e) for e in self.targets), self.expr)

    def type_check(self):
        actual = self.expr.type_check()
        for target in self.targets:
            expected = target.type_check()
            if expected != actual:
                raise BasicError(
                    "type error in assignment: can't assign {} {} to {}"
                    .format(actual, self.expr, target))

    def run(self, env):
        value = self.expr.evaluate(env)
        for target in self.targets:
            target.assign(env, value)


# Parsing

def tokenize(line):
    for m in re.finditer(TOKEN_RE, line):
        token = m.group(1)
        if not token.startswith(("'", "REM ")):
            yield m.group(1)


def parse_line(line):
    tokens = list(tokenize(line))

    point = 0

    def at_end():
        return point == len(tokens)

    def match(exact_token):
        nonlocal point
        if not at_end() and tokens[point] == exact_token:
            point += 1
            return True
        return False

    def require(exact_token):
        nonlocal point
        if at_end() or tokens[point] != exact_token:
            raise BasicError("expected " + exact_token)
        point += 1

    def require_identifier():
        nonlocal point
        if at_end() or not tokens[point][0].isalpha():
            raise BasicError("expected identifier")
        t = tokens[point]
        point += 1
        return t

    def parse_prim():
        nonlocal point
        if at_end():
            raise BasicError("expression expected")
        elif match('('):
            e = parse_expr()
            require(')')
            return e
        else:
            t = tokens[point]
            if t[0].isdigit():
                point += 1
                return NumberLiteralExpr(float(t))
            elif t[0] == '"':
                assert len(t) >= 2
                assert t[-1] == '"'
                point += 1
                return StringLiteralExpr(t[1:-1])
            elif t[0].isalpha():
                point += 1
                if t == 'RND':
                    return RndExpr()
                else:
                    return IdentifierExpr(t)
            else:
                raise BasicError("expected expression, got " + repr(t))

    def parse_call():
        nonlocal point
        if at_end():
            raise BasicError("expression expected")
        elif tokens[point][0].isalpha() and len(tokens) > point + 1 and tokens[point + 1] == '(':
            callee = tokens[point]
            point += 2
            if match(')'):
                return CallExpr(callee, [])
            args = [parse_expr()]
            while match(','):
                args.append(parse_expr())
            require(')')
            return CallExpr(callee, args)
        else:
            return parse_prim()

    def parse_term():
        e = parse_call()
        while True:
            for op in '*/':
                if match(op):
                    e = ArithmeticExpr(op, e, parse_prim())
                    break
            else:
                break
        return e

    def parse_arithmetic_expr():
        e = parse_term()
        while True:
            for op in '+-':
                if match(op):
                    e = ArithmeticExpr(op, e, parse_term())
                    break
            else:
                break
        return e

    def parse_expr():
        e = parse_arithmetic_expr()
        for op in ('<', '=', '>', '<>'):
            if match(op):
                return ComparisonExpr(op, e, parse_arithmetic_expr())
        return e

    def parse_lineno():
        nonlocal point
        if at_end() or not tokens[point].isdigit():
            raise BasicError("line number expected")
        lineno = int(tokens[point])
        point += 1
        return lineno

    if at_end():
        return EmptyStmt()
    elif match('STOP'):
        stmt = StopStmt()
    elif match('END'):
        stmt = EndStmt()
    elif match('DIM'):
        arr = tokens[point]
        if not arr[0].isalpha():
            raise BasicError("DIM expects array name")
        point += 1
        require('(')
        expr = parse_expr()
        require(')')
        stmt = DimStmt(arr, expr)
    elif match('IF'):
        condition = parse_expr()
        if not match('THEN'):
            raise BasicError("IF without THEN")
        dest = parse_lineno()
        stmt = IfStmt(condition, dest)
    elif match('FOR'):
        var = require_identifier()
        require('=')
        first = parse_expr()
        require('TO')
        last = parse_expr()
        return ForStmt(var, first, last)
    elif match('NEXT'):
        if at_end():
            return NextStmt()
        elif tokens[point][0].isalpha():
            var = tokens[point]
            point += 1
            return NextStmt(var)
    elif match('PRINT'):
        commas = 0
        while match(','):
            commas += 1  # not sure what leading commas are supposed to do
        exprs = []
        semicolon = False
        while not at_end():
            if match(';'):
                semicolon = True
                break
            elif match(','):
                exprs.append(PrintTab())
            else:
                exprs.append(parse_expr())
        return PrintStmt(exprs, commas, semicolon)
    elif match('LINPUT'):
        v = require_identifier()
        if not v.endswith('$'):
            raise BasicError("LINPUT requires string variable operand")
        return LinputStmt(v)
    elif match('ON'):
        e = parse_expr()
        require('GOTO')
        targets = [parse_lineno()]
        while match(','):
            targets.append(parse_lineno())
        stmt = OnGotoStmt(e, targets)
    elif match('GOTO'):
        target = parse_lineno()
        stmt = GotoStmt(target)
    elif match('GOSUB'):
        target = parse_lineno()
        stmt = GosubStmt(target)
    elif match('RETURN'):
        stmt = ReturnStmt()
    elif match('RANDOMIZE'):
        stmt = RandomizeStmt()
    elif tokens[point][0].isalpha():
        vars = [parse_arithmetic_expr()]
        while match('='):
            vars.append(parse_arithmetic_expr())
        value = vars.pop()
        for v in vars:
            if not isinstance(v, (IdentifierExpr, CallExpr)):
                raise BasicError("invalid assignment target")
        stmt = AssignmentStmt(vars, value)
    else:
        print(line)

    if not at_end():
        raise BasicError("stray tokens {} left at end of statement"
                         .format(repr(tokens[point:])))
    return stmt


class Program:
    def __init__(self, lines, line_table):
        self.lines = lines
        self.line_table = line_table

    @staticmethod
    def split_lines(line_iter, filename=""):
        lines = []
        for physical_lineno, line in enumerate(line_iter, start=1):
            line = line.strip()
            if line == '' or line.startswith("'"):
                continue
            first, rest = line.split(' ', 1)
            if first == 'REM':
                continue
            if first.isdigit():
                lineno = int(first)
                if lines and lines[-1][0] >= lineno:
                    raise ValueError(
                        "{}:{}: error: line number {} must be greater than previous line number {}"
                        .format(filename, physical_lineno, first, lines[-1][0]))
                yield lineno, rest.lstrip()

    @classmethod
    def from_lines(cls, line_iter, filename=""):
        lines = [(lineno, line) for lineno, line in cls.split_lines(line_iter, filename)]
        line_table = {lineno: index for index, (lineno, _) in enumerate(lines)}
        parsed_lines = []
        for index, (lineno, line) in enumerate(lines):
            try:
                stmt = parse_line(line)
                stmt.type_check()
                stmt.check_line_numbers(line_table)
            except BasicError as exc:
                print_line(lineno, line)
                raise
            parsed_lines.append((lineno, stmt))
        return cls(parsed_lines, line_table)

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            return cls.from_lines(f, filename)

    def all_jump_targets(self):
        all_targets = set()
        for _, stmt in self.lines:
            for target in stmt.jump_targets():
                all_targets.add(target)
        return all_targets


class Interpreter:
    def __init__(self, program, tracing=False):
        self.program = program
        self.status = 'run'
        self.variables = {}
        self.arrays = {}
        self.stack = []
        self._pc = 0
        self._jumped = False
        self.tracing = tracing
        self.output_column = 0

    def define_array(self, name, size):
        # The actual size of the array is size + 1 because in BASIC,
        # `DIM X(10)` defines an array that goes up to 10.
        self.arrays[name] = ["" if name.endswith('$') else 0.0] * (size + 1)

    def set_elem(self, name, index, value):
        if name not in self.arrays:
            self.define_array(name, 10)

        arr = self.arrays[name]
        if not (0 <= index < len(arr)):
            raise BasicError("array index out of range: {}({})".format(name, index))
        if self.tracing:
            print("*** Setting {}({}) to {}"
                  .format(name, index, basic_to_str(value)))
        arr[index] = value

    def jump(self, lineno):
        self.jump_to_index(self.program.line_table[lineno])

    def jump_to_index(self, index):
        self._pc = index
        self._jumped = True

    def get_next_index(self):
        return self._pc + 1

    def run(self):
        while self.status == 'run':
            pc = self._pc
            if self.tracing:
                print_line(*self.program.lines[pc])
            self._jumped = False
            try:
                self.program.lines[pc][1].run(self)
            except BasicError as exc:
                print_line(*self.program.lines[pc])
                raise exc
            if not self._jumped:
                self._pc += 1
                if self._pc >= len(self.program.lines):
                    self.status = 'stop'

def main():
    parser = argparse.ArgumentParser(description="Interpret some sort of BASIC program.")
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help="filename of the BASIC program to load and run")
    parser.add_argument('--trace', action='store_true',
                        help="enable tracing of every line executed and every value stored")
    options = parser.parse_args()
    [filename] = options.file
    program = Program.load(filename)
    Interpreter(program, tracing=options.trace).run()


if __name__ == '__main__':
    main()
