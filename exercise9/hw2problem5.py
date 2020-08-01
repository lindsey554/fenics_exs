from dolfin import *
import numpy as np
import matplotlib.pyplot as plt 

# Define parameters we might want to change
meshvec = [16,32,64,256]
pdeg = 1
seevec=[60,600,6000,60000,600000,6000000]
stop = 1e-10
step = 1.0
max_iter = 100

data_list = []
data_list.append([" ", int(16),int(32),int(64),int(256)])

for see in seevec:

	# Create list to add to list later
	data_pt = []
	data_pt.append(see)

	# Create a figure to put plots in later
	fig=plt.figure(figsize=(15,5))
	plt.suptitle('Newton\'s method for the Jeffrey-Hamel problem', size='xx-large', y=1.1)

	for meshsize in meshvec:

		# Create mesh and define function space
		#mesh = UnitSquareMesh(meshsize, meshsize) # if 2D
		mesh = UnitIntervalMesh(meshsize)
		V = FunctionSpace(mesh, "Lagrange", pdeg)

		# Define Dirichlet boundary (x = 0 or x = 1)
		def boundary(x):
			return x[0] < DOLFIN_EPS 

		# Define boundary condition
		u0 = Expression("0.0",degree=pdeg)
		bc = DirichletBC(V, u0, boundary)

		# Define some functions in the space
		u = TrialFunction(V)
		phi = TestFunction(V)
		U = Function(V)

		#f = Function(V) # if 2D
		f = Expression("C",C=see,degree=pdeg)

		# Define F as in class
		a = inner(grad(U), grad(phi))*dx+4.0*U*phi*dx
		n = 6.0*U*U*phi*dx
		F = a + n - f*phi*dx

		# Compute the Gateaux derivative/Jacobian
		G = derivative(F, U, u)

		# Make the initial U an interpolant of 1-x
		U0 = Expression("1.0", degree=pdeg)
		U.interpolate(U0)

		w = Function(V) # will solve for w
		newton_it = 0 # index iterations
		eps = 1.0 # just to start with

		# Newton iterations
		while eps > stop and newton_it < max_iter:

			newton_it += 1

			# Assemble matrix A from G and vector B from -F, and apply bc
			A, b = assemble_system(G, -F, bc)

			# solve Aw=b for w
			solve(A, w.vector(), b)

			# Update eps to be the 2-norm of w
			eps = np.linalg.norm(w.vector().get_local(), ord = 2)

			#fnorm = b.norm('l2') #check

			# U = U+U*step
			U.vector()[:] += step*w.vector()

			print('Iteration: ',newton_it,'\neps: ', eps,)
			#print('\nL2 norm of b: ', b.norm('l2'),) #check

		data_pt.append(newton_it)

		# Plot this solution
		ax1 = fig.add_subplot(1, 2, 1, title = 'Implemented')
		plot(U)
		ax1.set_ylim([0, 1.1*U.vector().norm('linf') ])
		plt.xlabel('x')
		plt.ylabel('u')

		# Solve using fenics
		problem = NonlinearVariationalProblem(F, U, bc, G)
		solver = NonlinearVariationalSolver(problem)
		solver.solve()

		# Plot fenics' solution
		ax2 = fig.add_subplot(1, 2, 2, title = 'Fenics')
		plot(U)
		ax2.set_ylim([0, 1.1*U.vector().norm('linf') ])
		plt.xlabel('x')
		plt.ylabel('u')

	# Display and then save figure
	plt.show()
	label = int(np.log10(see/(60)))
	filename = 'problem5_c'+str(label)+'.png'
	fig.savefig(filename, bbox_inches='tight')

	data_list.append(data_pt)

# Display results in a table
fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
table = ax.table(cellText=data_list, loc='center')
table.set_fontsize(14)
table.scale(1,1.5)
ax.axis('off')
plt.show()

# Save table as png
fig2.savefig("table5", bbox_inches='tight')
