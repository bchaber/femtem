# coding: utf-8
#
import numpy as np
import matplotlib.pyplot as plt
import separate_complex
from sys import argv
from export import *
from postprocessing import *
from emw import *
from dolfin import *

## Physical constants
freq = Constant(900e6)
omega = 2 * pi * freq
c = Constant(299792458.0)
mu0 = Constant(4 * pi * 1e-7)
eps0 = Constant(1.0 / (mu0 * c ** 2))
r0 = 1.5e-3
r1 = 4.75e-3
k0 = omega / c
Z0 = sqrt(mu0/eps0)

## Problem data
sigma_values = [0.00, 0.00]
epsr_values =  [1.00, 2.04]
mur_values =   [1.00, 1.00]

## Formulation
mesh_name = "meshes/coax-bent"

print("Reading %s" % mesh_name)
mesh = Mesh("%s.xml" % mesh_name)
n = FacetNormal(mesh)
subdomains = MeshFunction("size_t", mesh, "%s_physical_region.xml" % mesh_name)
boundaries = MeshFunction("size_t", mesh, "%s_facet_region.xml" % mesh_name)

# problem
V_re = FiniteElement("N2curl", mesh.ufl_cell(), 1)
V_im = FiniteElement("N2curl", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, V_re*V_im)
V0 = FunctionSpace(mesh, "DG", 0)

E_re, E_im = TrialFunctions(V)
T_re, T_im = TestFunctions(V)
T_im = -T_im

# boundaries
File(mesh_name + "_boundaries_original.pvd") << boundaries
File(mesh_name + "_subdomains_original.pvd") << subdomains
input_port = np.in1d(boundaries.array(), [3])
output_port = np.in1d(boundaries.array(), [38])
perfect_conductor = np.in1d(boundaries.array(), [1, 2, 4, 5] + [x for x in range(8, 38)] + [x for x in range(40, 46)])

# subdomains
ptfe = np.in1d(subdomains.array(), [1,2,3,4,5])
air  = np.in1d(subdomains.array(), [])

unassigned_boundary = 0
sbc_boundary = 1
pec_boundary = 2
pmc_boundary = 3
input_port_boundary = 4
output_port_boundary = 5

wpbc = True
lpbc = False

default_material = 0
air_material  = 0
ptfe_material = 1

boundaries.array()[:] = unassigned_boundary
boundaries.array()[perfect_conductor] = pec_boundary
boundaries.array()[input_port] = input_port_boundary
boundaries.array()[output_port] = pmc_boundary

subdomains.array()[:] = default_material
subdomains.array()[air] = air_material
subdomains.array()[ptfe] = ptfe_material

subdomain_markers = np.asarray(subdomains.array(), dtype='int32')

mur = Function(V0)
epsr = Function(V0)
sigma = Function(V0)
mur.vector()[:] = np.choose(subdomain_markers, mur_values)
epsr.vector()[:] = np.choose(subdomain_markers, epsr_values)
sigma.vector()[:] = np.choose(subdomain_markers, sigma_values)

k = k0 * sqrt( mur * epsr )

freq_v = freq([0,0,0]) * 1e-6
print("Compiling the form for freq = %d MHz" % freq_v)

print("Assembling Helmholtz equation")

dV = Measure('cell')
helmholtz_equation = HelmholtzEquation(T_re, T_im, E_re, E_im, k0, epsr, mur, dV)

if sbc_boundary in boundaries.array():
  print("Applying SBC")
  dS0 = Measure('exterior_facet')[boundaries](sbc_boundary)
  scattering_boundary = ScatteringBoundaryCondition(T_re, T_im, E_re, E_im, n, dS0)
  helmholtz_equation.add(scattering_boundary)

if input_port_boundary in boundaries.array():
  dSin = Measure('exterior_facet')[boundaries](input_port_boundary)

  if wpbc:
    print("Applying an INPUT WPBC")
    waveguide_port = WaveguidePortBoundaryCondition(T_re, T_im, E_re, E_im,
        n=n, k=k, mur=mur, r0=r0, r1=r1, E0=1.0, dS=dSin)    
    helmholtz_equation.add(waveguide_port)

  if lpbc:
    print("Applying an INPUT LPBC")
    lumped_port = LumpedPort(T_re, T_im, E_re, E_im,
         omega=omega, r0=r0, r1=r1, E0=1.0, dS=dSin)
    helmholtz_equation.add(lumped_port)

if output_port_boundary in boundaries.array():
  dSout = Measure('exterior_facet')[boundaries](output_port_boundary)

  if wpbc:
    print("Applying an output WPBC")
    waveguide_port = WaveguidePortBoundaryCondition(T_re, T_im, E_re, E_im,
        n=n, k=k, mur=mur, dS=dSin)    
    helmholtz_equation.add(waveguide_port)

  if lpbc:
    print("Applying an output LPBC")
    lumped_port = LumpedPort(T_re, T_im, E_re, E_im,
         omega=omega, dS=dSin)
    helmholtz_equation.add(lumped_port)

# Boundary conditions
pec = PerfectElectricConductor(mesh, degree=1)
bcs = [
	DirichletBC(V.sub(0), pec, boundaries, pec_boundary),
	DirichletBC(V.sub(1), pec, boundaries, pec_boundary),
]

print("Assembling of the system of equations...")

a, L = helmholtz_equation.forms()
A, b = assemble_system(a, L)
for bc in bcs:
  bc.apply(A)
  bc.apply(b)

E = Function(V, name="Electric Field")

print("Solving...")
solver = LUSolver(A, "mumps")
set_log_level(LogLevel.PROGRESS)
solver.solve(A, E.vector(), b)
print("...done")

E_re, E_im = E.split()

print("Postprocessing...")
f = ResultFile(mesh)
f.add_data(E_re, name='Ere (%d MHz)' % freq_v)
f.add_data(E_im, name='Eim (%d MHz)' % freq_v)
f.write("results/E-field.vtu")

File("results/subdomains.pvd") << subdomains
File("results/boundaries.pvd") << boundaries
File("results/epsr.pvd") << epsr
File("results/sigma.pvd") << sigma
File("results/mur.pvd") << mur

print("...done")
