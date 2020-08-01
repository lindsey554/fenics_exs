from dolfin import * 
from math import pi as pi
import matplotlib.pyplot as plt 
from matplotlib import ticker

# Define parameters we might want to change
meshsize = 32
pdeg = 2

# Create mesh and define function space
mesh = UnitSquareMesh(meshsize,meshsize)
V = FunctionSpace(mesh, "Lagrange", pdeg)

m = 0
mmax = 0
data = []

for n in range(65):

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=pdeg)
	alfa = 2**(-n)
	#print("iteration: ", n, "alpha:", alfa,)
	a = inner(grad(u), grad(v))*dx + alfa*u*v*ds
	L = (2*pi*pi)*f*v*dx

	# Compute solution 
	u = Function(V)
	solve(a == L, u)

	# Plot solutions, difference, and norm
	if n == 0 or n==2 or n==4 or n==8 or n==16 or n==32 or n==64:

		sol_and_err = []
		sol_and_err.append(u)
		sol_and_err.append(n)
		data.append(sol_and_err)
		m = m+1

	if n == 52: 
		mmax = m
		print("mmax:", mmax,)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=pdeg)
a = inner(grad(u), grad(v))*dx
L = (2*pi*pi)*f*v*dx

# Compute solution 
u = Function(V)
solve(a == L, u)

# Reset counter m to zero
m = 0

fig=plt.figure(figsize=(22,2.5))
plt.suptitle('Computed solutions as alpha goes to zero', size='xx-large', y=1.1)
plt.subplots_adjust(wspace=0.9, hspace = 0.2)

for data_pt in data: 

	ax1 = fig.add_subplot(1, mmax+2, m+1, title = 'alpha=2^'+str(-data_pt[1]))
	plt.xticks([0,0.5,1])
	plt.yticks([0,0.5,1])
	plt.set_cmap('jet')
	cb = plt.colorbar(plot(data_pt[0], interactive=True), pad=0.2, shrink = 0.8)
	#cb.ax.locator_params(nbins=5)

	m = m+1

ax1 = fig.add_subplot(1, mmax+2, mmax+2, title = 'alpha=0')
plt.xticks([0,0.5,1])
plt.yticks([0,0.5,1])
plt.set_cmap('jet')
cb = plt.colorbar(plot(u, interactive=True), pad=0.2, shrink = 0.8)

plt.show()

fig.savefig("figure5_1_zero", bbox_inches='tight')
