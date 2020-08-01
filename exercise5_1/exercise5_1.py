from dolfin import * 
from math import pi as pi
import matplotlib.pyplot as plt 
from matplotlib import ticker

# Create mesh and define function space
mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define exact solution to the Dirichlet problem
#ue = Expression("x[0]*x[1]*(1-x[1])", degree=2)
ue = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=2)
uze = interpolate(ue,V)

pm = 1
m = 0
mmax = 0
data = []

for n in range(53):

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=1)
	alfa = 2**( pm * n )
	#print("iteration: ", n, "alpha:", alfa,)
	a = inner(grad(u), grad(v))*dx + alfa*u*v*ds
	L = (2*pi*pi)*f*v*dx

	# Compute solution 
	u = Function(V)
	solve(a == L, u)

	# Compute norms
	l2_err = errornorm(u, uze, norm_type='L2', degree_rise=3)
	print("error:", l2_err,)

	# Plot solutions, difference, and norm
	if n == 0 or n==2 or n==4 or n==8 or n==16 or n==32 or n==52 or n==10 or n==12:

		sol_and_err = []
		sol_and_err.append(u)
		sol_and_err.append(l2_err)
		sol_and_err.append(n)
		data.append(sol_and_err)
		print("saved solution:", n,)
		m = m+1

	if n == 52: 
		mmax = m
		print("mmax:", mmax,)

# Reset counter m to zero
m = 0

if pm == 1:

	fig=plt.figure(figsize=(26,4))
	plt.suptitle('Computed solutions as alpha = 2^n goes to infinity', size='xx-large', y=1.1)
	plt.subplots_adjust(wspace=1.1, hspace = 0.2)

	for data_pt in data: 

		ax1 = fig.add_subplot(2, mmax+1, m+1, title = 'n='+str(pm * data_pt[2]))
		plt.xticks([0,0.5,1])
		plt.yticks([0,0.5,1])
		plt.set_cmap('jet')
		cb = plt.colorbar(plot(data_pt[0], interactive=True), pad=0.2, shrink = 0.8)
		#cb.ax.locator_params(nbins=5)

		ax2 = fig.add_subplot(2, mmax+1, mmax+m+2, title = 'difference')
		plt.xticks([0,0.5,1])
		plt.yticks([0,0.5,1])
		plt.set_cmap('spring')
		plt.colorbar(plot(data_pt[0]-uze, interactive=True), pad=0.3, shrink = 0.8)

		m = m+1

	ax1 = fig.add_subplot(2, mmax+1, mmax+1, title = 'Dirichlet')
	plt.xticks([0,0.5,1])
	plt.yticks([0,0.5,1])
	plt.set_cmap('jet')
	cb = plt.colorbar(plot(uze, interactive=True), pad=0.3, shrink = 0.8)
	#cb.ax.locator_params(nbins=5)

	plt.show()
	fig.savefig("figure5_1_infty", bbox_inches='tight')

if pm == -1: 

	fig=plt.figure(figsize=(22,2.5))
	plt.suptitle('Computed solutions as alpha = 2^n goes to zero', size='xx-large', y=1.1)
	plt.subplots_adjust(wspace=0.9, hspace = 0.2)

	for data_pt in data: 

		ax1 = fig.add_subplot(1, mmax+1, m+1, title = 'n='+str(pm * data_pt[2]))
		plt.xticks([0,0.5,1])
		plt.yticks([0,0.5,1])
		plt.set_cmap('jet')
		cb = plt.colorbar(plot(data_pt[0], interactive=True), pad=0.2, shrink = 0.8)
		#cb.ax.locator_params(nbins=5)

		m = m+1

	plt.show()

	fig.savefig("figure5_1_zero", bbox_inches='tight')
