from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
you = Expression("(sin(3.141592*x[0]))*(sin(3.141592*x[1]))", degree=2)
a = inner(grad(u), grad(v))*dx
L = (2*3.141592*3.141592)*you*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

import matplotlib.pyplot as plt
c=plot(u, interactive=True)
plt.colorbar(c)
plt.show()
