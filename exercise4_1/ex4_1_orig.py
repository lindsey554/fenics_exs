from fenics import *
from math import pi as pi
from math import log2 as log2
from timeit import default_timer as timer

startime=timer()
#meshsize=int(sys.argv[1])
#pdeg=int(sys.argv[2])
pvec     = [1, 2, 3, 4, 5, 6]
meshvec  = [8,16,32,64,128,256]
p_err = 1.0 ## placeholder

#for pdeg in pvec:
for meshsize in meshvec:
  #meshsize = meshvec[0] 
  pdeg = pvec[2]
  # Create mesh and define function space
  mesh = UnitSquareMesh(meshsize, meshsize)
  V = FunctionSpace(mesh, "Lagrange", pdeg)
  
  # Define Dirichlet boundary (x = 0 or x = 1 or y = 0 or y = 1)
  def boundary(x):
      return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
  
  # Define boundary condition
  u0 = Constant(0.0)
  bc = DirichletBC(V, u0, boundary)
  
  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Expression("(sin(mypi*x[0]))*(sin(mypi*x[1]))",mypi=pi,degree=pdeg+3,quadrature_degree=5)
  a = inner(grad(u), grad(v))*dx
  L = (2*pi*pi)*f*v*dx 
  
  # Compute solution
  u = Function(V)
  solve(a == L, u, bc)
  aftersolveT=timer()
  totime=aftersolveT-startime

  l2err = errornorm(f,u,norm_type='l2',degree_rise=3)
  erate  = log2(p_err/l2err)
  print("deg: ",pdeg,", meshsize h:%.3e"%(1.0/meshsize),", l2 error: %.2e"%l2err,", time:%.3f"%totime,"sec, rate: %.2e"%erate)
  p_err = l2err
