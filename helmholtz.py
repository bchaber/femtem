from dolfin import *
from separate_complex import *

mu0 = 4*pi*1e-7
eps0 = 8.85e-12
epsr, mur = 2, 1
c = Constant(2.99792e8)
f = Constant(800e6)
k = Constant(2*pi*f/c*sqrt(epsr*mur))
mesh = UnitSquareMesh(50,50)
V_re = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
V_im = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, V_re*V_im)
dV = Measure('cell', domain=mesh)
dS = Measure('exterior_facet', domain=mesh)
T_re, T_im = TestFunctions(V)
Ez_re, Ez_im = TrialFunctions(V)

a_re, a_im = separate("inner(grad(T),grad(Ez)) - k*k*(T*Ez)",
        k=k,Ez=(Ez_re,Ez_im),T={'re':T_re,'im':T_im})
sbc_re, sbc_im = separate("-1j*k*(T*Ez)",
        k=k,Ez=(Ez_re,Ez_im),T={'re':T_re,'im':T_im})
class ComplexPointSource(UserExpression):
    def eval(self, value, x):
        if near(x[0], 0.5) and near(x[1], 0.5):
            value[0] = 1.0
        else:
            value[0] = 0.0
        value[1] = 0.0
    def value_shape(self):
        return (2,)

zero = Constant(0.0)
f_re, f_im = ComplexPointSource(degree=1)
a = (a_re + a_im)*dV + (sbc_re + sbc_im)*dS
L = T_re*f_re*dV + T_im*f_im*dV
A = assemble(a)
b = assemble(L)
pec_re = DirichletBC(V.sub(0), zero, lambda x: near(x[0], 0.0))
pec_im = DirichletBC(V.sub(1), zero, lambda x: near(x[0], 0.0))
pec_re.apply(A,b)
pec_im.apply(A,b)
from dolfin import Function
Ez = Function(V)
solve(A, Ez.vector(), b, 'mumps')
Ez_re, Ez_im = Ez.split()
plot(Ez_re)
import matplotlib.pyplot as plt
plt.show()
