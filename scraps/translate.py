import basic
from basic import Program, print_line
import argparse


BREAK_STMT_TYPES = (basic.GotoStmt, basic.OnGotoStmt, basic.ReturnStmt, basic.StopStmt, basic.EndStmt)

def should_break_after(stmt):
    return stmt is not None and isinstance(stmt[1], BREAK_STMT_TYPES)

def should_break_before(targets, stmt):
    return stmt[0] in targets



def translate_expr(expr):
    if isinstance(expr, basic.IdentifierExpr):
        return "BASIC variable {}".format(expr.name)
    else:
        return str(expr)

def main():
    parser = argparse.ArgumentParser(description="Translate some sort of BASIC program to Inform 7.")
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help="filename of the BASIC program to translate")
    options = parser.parse_args()
    [filename] = options.file
    program = Program.load(filename)
    ProgramTranslator(program).translate()

class ProgramTranslator:
    def __init__(self, program):
        self.program = program
        self.targets = program.all_jump_targets()
        self.cached_print_matter = ''

    def translate(self):
        program = self.program
        targets = self.targets
        for a, b in zip([None] + program.lines[:-1], program.lines):
            if should_break_after(a) or should_break_before(targets, b):
                self.flush_prints()
                print()

            lineno, stmt = b
            if lineno in targets:
                print("This is the line {} rule:".format(lineno))

            self.translate_stmt(stmt)


    def translate_stmt(self, stmt):
        if isinstance(stmt, basic.PrintStmt):
            self.cache_print(stmt)
            return
        else:
            self.flush_prints()

        if isinstance(stmt, (basic.StopStmt, basic.EndStmt)):
            print("\tend the story finally.")
        elif isinstance(stmt, basic.GotoStmt):
            print("\tcontinue with the line {} rule.".format(stmt.target))
        elif isinstance(stmt, basic.GosubStmt):
            print("\tfollow the line {} rule;".format(stmt.target))
        elif isinstance(stmt, basic.LinputStmt):
            print("\tget a line of input;")
            print("\tnow BASIC variable {} is the user input;".format(stmt.var))
        elif isinstance(stmt, basic.EmptyStmt):
            pass
        elif isinstance(stmt, basic.AssignmentStmt) and len(stmt.targets) == 1:
            print('\tnow {} is {};'.format(translate_expr(stmt.targets[0]), translate_expr(stmt.expr)))
        else:
            print("\t" + str(stmt))

    def cache_print(self, stmt):
        assert isinstance(stmt, basic.PrintStmt)
        s = ''
        for expr in stmt.exprs:
            if isinstance(expr, basic.StringLiteralExpr):
                s += expr.value
            elif isinstance(expr, basic.PrintTab):
                s += "[tab]"
            else:
                s += "[" + translate_expr(expr) + "]"
        if not stmt.trailing_semicolon:
            s += "[line break]"
        self.cached_print_matter += s

    def flush_prints(self):
        if self.cached_print_matter:
            print('\tprint "{}";'.format(self.cached_print_matter))
            self.cached_print_matter = ''


if __name__ == '__main__':
    main()
