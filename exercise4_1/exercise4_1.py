from fenics import *
from math import pi as pi
from math import log2 as log2
from timeit import default_timer as timer
import matplotlib.pyplot as plt 
import argparse

# Define command line argument options
parser = argparse.ArgumentParser(description=
	'Computes solutions to the variational problem \
	and outputs a table of error and timing data.')
parser.add_argument('norm', type=int, choices=[1, 2],
	help='type "1" for H1 norm, "2" for L2 norm')
parser.add_argument("-m", "--meshnumber", type=int, 
	choices=[1, 2, 3, 4, 5, 6, 7, 8],
	help='ex: type "-m 3" to set mesh number equal to 2^3=8.')
parser.add_argument("-d", "--degree", type=int, 
	choices=[1, 2, 3, 4, 5, 6, 7, 8] ,
	help='ex: type "-d 3" to set polynomial degree equal to 3.')

# Parse command line arguments
args = parser.parse_args()

# Create list to hold data
data_list = []

# Use arguments or set to defaults
if args.norm == 1: 
	data_list.append(["degree", "mesh number", "H1 error",
		"time (s)", "rate"])
else:
	data_list.append(["degree", "mesh number", "L2 error",
		"time (s)", "rate"])

if args.meshnumber is not None:
	meshvec = [2**args.meshnumber]
else: 
	meshvec  = [8,16,32,64,128,256]

if args.degree is not None:
	pvec = [args.degree]
else:
	pvec     = [1, 2, 3, 4, 5, 6]

# Set initial values
p_err = 1.0 
erate = 1.0 

# Start timer
startime=timer()

# Iterate through chosen degree and mesh sizes
for pdeg in pvec:
	print("degree: " ,pdeg,)
	for meshsize in meshvec:

		# Continue past combos that are too big
		if pdeg>=5 and meshsize==256: 
			continue
		if pdeg>=6 and meshsize==64: 
			continue

		# Create mesh and define function space
		mesh = UnitSquareMesh(meshsize, meshsize)
		V = FunctionSpace(mesh, "Lagrange", pdeg)
	  
		# Define Dirichlet boundary
		#def boundary(x):
		#	return x[0] < DOLFIN_EPS \
		#		or x[0] > 1.0 - DOLFIN_EPS \
		#		or x[1] < DOLFIN_EPS \
		#		or x[1] > 1.0 - DOLFIN_EPS
	  
		# Define boundary condition
		u0 = Constant(0.0)
		bc = DirichletBC(V, u0, DomainBoundary())
	  
		# Define variational problem
		u = TrialFunction(V)
		v = TestFunction(V)

		# x(1-x)y(1-y)
		f = Expression("x[0]*(1-x[0])*x[1]*(1-x[1])",
			mypi=pi,degree=pdeg+3,quadrature_degree=5)

		a = inner(grad(u), grad(v))*dx

		# 2(x(x-1)+(y-1)y)
		l = Expression("2.0*(x[0]*(1-x[0])+x[1]*(1-x[1]))",
			mypi=pi,degree=pdeg+3,quadrature_degree=5)
		L = l*v*dx

		# Compute solution
		u = Function(V)
		solve(a == L, u, bc)
		aftersolveT=timer()
		totime=aftersolveT-startime
		
		if args.norm == 2:
			# Compute l2 error
			l2_err = errornorm(f,u,norm_type='l2',degree_rise=3)
			l2_erate  = log2(p_err/l2_err)
			p_err = l2_err
			erate=l2_erate

		else:
			# Compute H1 error
			h1_err = errornorm(f,u,norm_type='H1',degree_rise=3)
			h1_erate = log2(p_err/h1_err)
			p_err = h1_err
			erate=h1_erate

		# Append this data to the list of data
		data_list.append([int(pdeg),int(meshsize),
			"%.2e"%p_err,"%.3f"%totime,"%.2e"%erate])

# Display results in a table
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
table = ax.table(cellText=data_list, loc='center')
table.set_fontsize(14)
table.scale(1,1.5)
ax.axis('off')
plt.show()

# Save table as png
if args.norm == 2:
	fig.savefig("figure4_1", bbox_inches='tight')
else:
	fig.savefig("figure4_2", bbox_inches='tight')
