from sympy.printing.python import PythonPrinter, StrPrinter
from sympy import Symbol, Function, symbols, re, im, S

class ComplexFormPrinter(PythonPrinter):
    def __init__(self, skip_multiplication_by_one=False):
        self.skip_multiplication_by_one = skip_multiplication_by_one
        PythonPrinter.__init__(self)

    def _print_Float(self,expr):
        return str(expr)

    def _print_Function(self, expr):
        if expr.func == re:
            arg = expr.args[0]
            if hasattr(arg, 'name'):
                return arg.name + '_re'
            return str(arg.func) + '(' + ','.join([self._print(x) for x in arg.args]) + ')'
        if expr.func == im:
            arg = expr.args[0]
            if hasattr(arg, 'name'):
                return arg.name + '_im'
            return str(arg.func) + '(' + ','.join([self._print(x) for x in arg.args]) + ')'
        return PythonPrinter._print_Function(self, expr)

    def _print_Symbol(self, expr):
        return expr.name

    def _print_Rational(self,expr):
        return StrPrinter._print_Rational(self,expr)

    def _print_Mul(self,expr):
        def rearrange_multiplication(expr):
            sign = ''
            if expr.startswith('-'):
                expr = expr[1:]
                sign = '-'
            measures = []
            functions = []
            numbers = []
            symbols = []

            args = expr.split('*')
            for arg in args:
                if arg.startswith('d'):
                    measures.append(arg)
                elif arg.find('('):
                    functions.append(arg)
                elif arg.find('.'):
                    numbers.append(arg)
                else:
                    symbols.append(arg)
            return sign + '*'.join(numbers + symbols + functions + measures)
        return rearrange_multiplication(StrPrinter._print_Mul(self, expr)).replace('.00000000000000','')

class ComplexFunction(Function):
    @classmethod
    def eval(cls, *args):
        for arg in args:
            if arg.is_Number and arg is S.Zero:
                return S.Zero

    def _eval_is_real(self):
        for arg in self.args:
            if not arg.is_real:
                return False
        return True
        
class grad(ComplexFunction):
    nargs = 1
    def as_real_imag(self, deep=True, **hints):
        u_re, u_im = self.args[0].as_real_imag()
        return (grad(u_re), grad(u_im))

class div(grad):
    nargs = 1
    def as_real_imag(self, deep=True, **hints):
        u_re, u_im = self.args[0].as_real_imag()
        return (div(u_re), div(u_im))

class curl(grad):
    nargs = 1
    def as_real_imag(self, deep=True, **hints):
        u_re, u_im = self.args[0].as_real_imag()
        return (curl(u_re), curl(u_im))

class nabla_grad(grad):
    nargs = 1
    def as_real_imag(self, deep=True, **hints):
        u_re, u_im = self.args[0].as_real_imag()
        return (nabla_grad(u_re), nabla_grad(u_im))

class rot(curl):
    pass

class inner(ComplexFunction):
    nargs = 2
    treat_arguments_as_vectors = True

    @classmethod
    def eval(cls, u, v):
        if not cls.treat_arguments_as_vectors and u.is_real and v.is_real:
            return u*v
        return ComplexFunction.eval(cls, u, v)

    def as_real_imag(self, deep=True, **hints):
        u_re,u_im = self.args[0].as_real_imag()
        v_re,v_im = self.args[1].as_real_imag()
        return (inner(u_re,v_re) - inner(u_im,v_im), inner(u_re,v_im) + inner(v_re,u_im))

class cross(ComplexFunction):
    nargs = 2
    def as_real_imag(self, deep=True, **hints):
        u_re,u_im = self.args[0].as_real_imag()
        v_re,v_im = self.args[1].as_real_imag()
        return (cross(u_re,v_re) - cross(u_im,v_im), cross(u_re,v_im) + cross(v_re,u_im))

def separate(form_as_str, **kwargs):
    for var_name, var in kwargs.items():
        if type(var) is tuple: # as complex value: (real, imaginary)
            locals()[var_name] = symbols(var_name)
        else: # as real value: value
            locals()[var_name] = symbols(var_name, real=True)

    form_printer = ComplexFormPrinter()
    form_as_sympy = eval(form_as_str)
    expanded_expression = form_as_sympy.expand(complex=True)
    real_part_as_sympy, imag_part_as_sympy = expanded_expression.as_real_imag()
    real_part_as_str = form_printer.doprint(real_part_as_sympy)
    imag_part_as_str = form_printer.doprint(imag_part_as_sympy)

    from dolfin import cross, curl, grad, inner, Function

    for var_name, var in kwargs.items():
        if type(var) is tuple: # as complex value: (real, imaginary)
            locals()[var_name + '_re'] = var[0]
            locals()[var_name + '_im'] = var[1]
        elif type(var) is dict:
            locals()[var_name + '_re'] = var['re']
            locals()[var_name + '_im'] = var['im']
            real_part_as_str = real_part_as_str.replace(var_name, var_name + '_re')
            imag_part_as_str = imag_part_as_str.replace(var_name, var_name + '_im')
        else: # as real value: value
            locals()[var_name] = var

    print("Debug: real part of the expression after separation:\n\t'%s'" % real_part_as_str)
    print("Debug: imaginary part of the expression after separation:\n\t'%s'" % imag_part_as_str)
    real_part_as_ufl = eval(real_part_as_str)
    imaginary_part_as_ufl = eval(imag_part_as_str)
    return real_part_as_ufl, imaginary_part_as_ufl

if __name__ == '__main__':
    a_re, a_im = separate("T*E", T={'re':'T_re','im':'T_im'}, E=('E_re','E_im'))
