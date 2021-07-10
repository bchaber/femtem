from dolfin import *

try:
    from vtk import vtkUnstructuredGrid, vtkPolyData, vtkXMLUnstructuredGridWriter, vtkXMLUnstructuredGridReader, vtkXMLPolyDataWriter, vtkXMLPolyDataReader
    from vtk import vtkPoints, vtkTetra, vtkVertex, VTK_TETRA, vtkCellArray, vtkDoubleArray
except ImportError:
    print("VTK not available")
import math
import os.path
import numpy as np
import numpy.linalg
import scipy.integrate
import scipy.sparse
import scipy.io
import separate_complex

class ComplexForm(object):
    def __init__(self, T_re, T_im):
        zero = Constant((0.0, 0.0, 0.0))
        self.re = inner(T_re, zero)*dV
        self.im = inner(T_im, zero)*dV

    def add(self, another):
        self.re += another.re
        self.im += another.im

    def forms(self):
        combined = self.re + self.im
        return lhs(combined), rhs(combined)

class HelmholtzEquation(ComplexForm):
   def __init__(self, T_re, T_im, E_re, E_im, k0, epsr, mur, dV):
        zero = Constant((0.0, 0.0, 0.0))
        helmholtz_form = """inner(curl(T), invmur*curl(E))*dV """ + \
                         """- k0**2*inner(T, epsr*E)*dV"""
        self.re, _ = separate_complex.separate(helmholtz_form,
            E=(E_re, E_im), T=T_re, epsr=epsr, invmur=1./mur, k0=k0, dV=dV)
        _, self.im = separate_complex.separate(helmholtz_form,
            E=(E_re, E_im), T=T_im, epsr=epsr, invmur=1./mur, k0=k0, dV=dV)
        self.re += inner(T_re, zero)*dV
        self.im += inner(T_im, zero)*dV
        print("id %d" % id(self))

class ScatteringBoundaryCondition(ComplexForm):
    def __init__(self, T_re, T_im, E_re, E_im, n, k, dS):
        sbc_form = """1j*k*inner(cross(n, T), cross(n, E))*dS"""
        self.re, _ = separate_complex.separate(sbc_form,
            E=(E_re, E_im), T=T_re, n=n, k=k, dS=dS)
        _, self.im = separate_complex.separate(sbc_form,
            E=(E_re, E_im), T=T_im, n=n, k=k, dS=dS)

class PerfectElectricConductor(UserExpression):
    def eval_cell(self, values, x, ufc_cell):
        values[:] = 0.0

    def value_shape(self):
        return (3,)

class PerfectMagneticConductor(ComplexForm):
    pass

class ModeCoaxialWaveguideTEM(UserExpression):
    def setup(self, R0, R1, A=0, c=(0.0, 0.0, 0.0)):
        self._hash = None
        self.A = A
        self.B = 1/np.log(R1/R0)
        self.c = np.array(c)
        return self

    def eval_cell(self, values, x, ufc_cell):
        self.eval(values, x)

    def eval(self, values, x):
        A = self.A
        B = self.B
        c = self.c

        r = (x - c)
        R = np.linalg.norm(r)
        r /= R

        values[:] =  A * B * r/R

    def value_shape(self):
        return (3,)

    def __repr__(self):
        return 'ModeCoaxialWaveguideTEM(%d)' % (self.A)
class WaveguidePortBoundaryCondition(ComplexForm):
    def __init__(self, T_re, T_im, E_re, E_im, n, k, mur, r0, r1, dS, E0=0,c=(0.0, 0.0, 0.0)):
        wpbc_form = """1j*inner(cross(n, T), k*invmur*cross(n, E))*dS """ + \
                    """- 2j*inner(T, k*invmur*e0TEM)*dS"""

        e0TEM = Expression(("0.0", "0.0", "0.0"), degree=1)
        if E0 > 0.0:
            e0TEM = ModeCoaxialWaveguideTEM(degree=2).setup(r0, r1, E0, c=c)

        self.re, _ = separate_complex.separate(wpbc_form,
            E=(E_re, E_im), T=T_re, n=n, k=k, invmur=1./mur, e0TEM=e0TEM, dS=dS)
        _, self.im = separate_complex.separate(wpbc_form,
            E=(E_re, E_im), T=T_im, n=n, k=k, invmur=1./mur, e0TEM=e0TEM, dS=dS)

class LumpedPortBoundaryCondition(ComplexForm):
    def __init__(self, T_re, T_im, E_re, E_im, omega, mur, r0, r1, dS, E0):
        lpbc_form = """-1j*inner(omega*(E-2*e0TEM), T)*inveta*dS"""

        Zref = 50 # Ohm
        etaref = 2*pi*Zref/1.1527

        e0TEM = Expression(("0.0", "0.0", "0.0"), degree=1)
        if E0 > 0.0:
            e0TEM = ModeCoaxialWaveguideTEM(degree=2).setup(r0, r1, E0, c=(0.0, 0.0, 0.0))

        self.re, _ = separate_complex.separate(lpbc_form,
            E=(E_re, E_im), T=T_re, omega=omega, inveta=1./etaref, e0TEM=e0TEM, dS=dS)
        _, self.im = separate_complex.separate(lpbc_form,
            E=(E_re, E_im), T=T_im, omega=omega, inveta=1./etaref, e0TEM=e0TEM, dS=dS)
