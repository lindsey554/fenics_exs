from dolfin import * 
from math import pi as pi
import matplotlib.pyplot as plt 

# Create mesh and define function space
mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=1)
alfa = 1.0
a = inner(grad(u), grad(v))*dx + alfa*u*v*ds
L = (2*pi*pi)*f*v*dx

# Compute solution 
u = Function(V)
solve(a == L, u)

# Plot solution
c=plot(u, interactive=True)
plt.colorbar(c)
plt.show()
