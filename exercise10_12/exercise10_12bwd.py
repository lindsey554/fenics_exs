import sys, math
from dolfin import *
import time
import matplotlib.pyplot as plt
from numpy import ceil as ceil

#k=float(sys.argv[1])
#deg=int(sys.argv[2])
#mno=int(sys.argv[3])

k  = 0.01  ## timestep
deg = 1   ## Lagrange degree
mno = 5 ## number of x-subintervals
omega = 5 ## frequency of rhs function f
Tfin  = 0.1 ## final time
# -- number of timesteps
nsteps = Tfin/k

# Create mesh and define function space
mesh = UnitIntervalMesh(mno)
V = FunctionSpace(mesh, "Lagrange", deg)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
# g = Expression("0.5-std::abs(x[0]-0.5)",degree=deg)
g = Expression("x[0]*(1-x[0])",degree=deg)
f = Expression('sin(omega*pi*x[0])',omega = omega, degree=deg)
u0 = Expression("0",degree=deg)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
uold = Function(V)
v = TestFunction(V)
a = u*v*dx
F = uold*v*dx - k*inner(grad(uold),grad(v))*dx + k*f*v*dx
u = Function(V)
uold.interpolate(g)
n=0 ## n = number of timesteps
u.assign(uold)

while k*n < Tfin:
	n = n+1
	# Compute one time step
	# Compute one time step
	solve(a == F, u, bc)
	uold.assign(u)
	plot(u)
	tstr = 'u_heat_fe_'+str(k)+'_'+str(Tfin)+'.png'
	plt.savefig(tstr,dpi=300)
