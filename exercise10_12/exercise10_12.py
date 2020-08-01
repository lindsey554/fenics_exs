import sys, math
from dolfin import *
import time
import matplotlib.pyplot as plt
from numpy import ceil as ceil

#k=float(sys.argv[1])
#deg=int(sys.argv[2])
#mno=int(sys.argv[3])

k  = 0.001  ## timestep
deg = 1   ## Lagrange degree
mno = 128 ## number of x-subintervals
omega = 5 ## frequency of rhs function f
Tfin  = 10 ## final time
# -- number of timesteps
nsteps = Tfin/k

# Create mesh and define function space
mesh = UnitIntervalMesh(mno)
V = FunctionSpace(mesh, "Lagrange", deg)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
	return x[0] < DOLFIN_EPS
	#return x[0] < DOLFIN_EPS or \
	#	x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
g = Expression("x[0]*(1-x[0])",degree=deg)
#g = Expression("x[0]*(2-x[0])",degree=deg) #test
#g = Expression("x[0]*(3-x[0]*x[0])",degree=deg) #test
f = Expression('sin(omega*pi*x[0])',omega = omega, degree=deg)

#f = Constant(0.0) 
u0 = Expression("0",degree=deg)

bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
uold = Function(V)
v = TestFunction(V)
a = k*inner(grad(u), grad(v))*dx + u*v*dx 
F = uold*v*dx + k*f*v*dx
u = Function(V)

# uold starts as the initial condition g
uold.interpolate(g)

n=0 ## n = number of timesteps
while k*n < Tfin:

	n = n+1
	#Compute one time step
	solve(a == F, u, bc)
	uold.assign(u)
	plot(u)
	
tstr = 'u_heat_be_'+str(k)+'_'+str(Tfin)+'.png'
plt.savefig(tstr,dpi=300)
