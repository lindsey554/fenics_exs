from dolfin import *
from math import pi as pi
import matplotlib.pyplot as plt

def boundary(x):
	return x[0] < DOLFIN_EPS or \
		x[0] > 1.0 - DOLFIN_EPS or \
		x[1] < DOLFIN_EPS or \
		x[1] > 1.0 - DOLFIN_EPS

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u0 = Constant(0.0)
bc_v1 = DirichletBC(V, u0, "on_boundary")
bc_v2 = DirichletBC(V, u0, DomainBoundary())
bc_v3 = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
you = Expression("(sin(pi*x[0]))*(sin(pi*x[1]))", degree=2)
a = inner(grad(u), grad(v))*dx
L = (2*pi**2)*you*v*dx

# Compute solution
u1 = Function(V)
u2 = Function(V)
u3 = Function(V)

solve(a == L, u1, bc_v1)
solve(a == L, u2, bc_v2)
solve(a == L, u3, bc_v3)

# Compute norms
h1_err1 = errornorm(u1, u2, norm_type='H1', degree_rise=3)
h1_err2 = errornorm(u1, u3, norm_type='H1', degree_rise=3)

# Plot solutions
fig=plt.figure(figsize=(16,4)) 
plt.suptitle('Original method vs two others: H1 error='\
	+"%.15e"%h1_err1+', '\
	+"%.15e"%h1_err2, \
	size='xx-large', y=1.1)

plt.subplot(131, title='using "on boundary"')
plt.colorbar(plot(u1, interactive=True))

plt.subplot(132, title='using DomainBoundary()')
plt.colorbar(plot(u2, interactive=True))

plt.subplot(133, title='using boundary')
plt.colorbar(plot(u3, interactive=True))

plt.show()

# Save plot as png
fig.savefig("figure2_3", bbox_inches='tight')
