from dolfin import *
import numpy as np
import matplotlib.pyplot as plt 

# Define parameters we might want to change
meshsize = 32
pdeg = 2

# Create mesh and define function space
mesh = UnitSquareMesh(meshsize, meshsize)
V = FunctionSpace(mesh, "Lagrange", pdeg)

# Dirichlet boundary (x=0 or y=0 or y=1)
def boundary(x):
	return x[0] < DOLFIN_EPS or \
		x[1] < DOLFIN_EPS or \
		x[1] > 1 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define some functions in the space
u = TrialFunction(V)
v = TestFunction(V)
U = Function(V)

# Define F (an expression) like in class
#k = 1.0*U+1.0
#F = inner(k*grad(U), grad(v))*dx
F = inner((1.0*U+1.0)*grad(U), grad(v))*dx

# Compute the Jacobian
J = derivative(F, U, u)

# Make the initial U an interpolant of 1-x
U0 = Expression("1-x[0]", degree=2)
U.interpolate(U0)

if False:
	problem = NonlinearVariationalProblem(F, U, bc, J)
	solver = NonlinearVariationalSolver(problem)
	solver.solve()
	print('got False')
else:

	U_inc = Function(V)

	eps = 1
	a_tol, r_tol = 1e-7, 1e-10
	newton_it = 0
	step = 1.0

	# Newton iterations
	while eps > 1e-10 and newton_it < 10:

		# Increment counter
		newton_it += 1

		# Assemble system and apply bc
		A, B = assemble_system(J, -F, bc)

		# Determine step direction
		solve(A, U_inc.vector(), B)
		eps = np.linalg.norm(U_inc.vector().get_local(), ord = 2)

		# Assemble and apply bcs to the matrix M
		M = assemble(F)
		bc.apply(M)

		print('\nL2 norm of B: ', B.norm('l2'), '\nnorm of M: ', np.linalg.norm(M.get_local(), ord = 2) )
		fnorm = B.norm('l2')

		# U= U+U*step
		U.vector()[:] += step*U_inc.vector()    # New u vector

		print('Iteration: ',newton_it,'\neps: ', eps, '\nfnorm: ',fnorm,)

		plot(U)
		plt.show()

plot(U)
plt.show()
