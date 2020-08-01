from dolfin import * 
from math import pi as pi
import matplotlib.pyplot as plt 
from matplotlib import ticker

# Define parameters we might want to change
meshsize = 64
pdeg = 2

# Create mesh and define function space
mesh = UnitSquareMesh(meshsize,meshsize)
V = FunctionSpace(mesh, "Lagrange", pdeg)

# Define exact solution to the Dirichlet problem
ue = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=pdeg)
uze = interpolate(ue,V)

m = 0
mmax = 0

data = []
err_data = []
err_data.append(["Value of alpha", "L2 error norm", "H1 error norm"])

for n in range(53):

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=pdeg)
	alfa = 2**n
	#print("iteration: ", n, "alpha:", alfa,)
	a = inner(grad(u), grad(v))*dx + alfa*u*v*ds
	L = (2*pi*pi)*f*v*dx

	# Compute solution 
	u = Function(V)
	solve(a == L, u)

	# Compute norms
	l2_err = errornorm(u, uze, norm_type='L2')
	h1_err = errornorm(u, uze, norm_type='H1')
	err_data.append([int(alfa),"%.2e"%l2_err,"%.2e"%h1_err])
	print("error:", l2_err,)

	# Plot solutions, difference, and norm
	if n == 0 or n==2 or n==4 or n==8 or n==16 or n==32 or n==52:

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

fig=plt.figure(figsize=(26,4))
plt.suptitle('Computed solutions as alpha = 2^n goes to infinity', size='xx-large', y=1.1)
plt.subplots_adjust(wspace=1.1, hspace = 0.2)

for data_pt in data: 

	ax1 = fig.add_subplot(2, mmax+1, m+1, title = 'n='+str(data_pt[2]))
	plt.xticks([0,0.5,1])
	plt.yticks([0,0.5,1])
	plt.set_cmap('jet')
	cb = plt.colorbar(plot(data_pt[0], interactive=True), pad=0.2, shrink = 0.8)

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

plt.show()
fig.savefig("figure5_1_infty", bbox_inches='tight')

# Display error results in a table
fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
table = ax.table(cellText=err_data, loc='center')
table.set_fontsize(14)
table.scale(1,1.5)
ax.axis('off')
plt.show()

# Save table as png
fig2.savefig("table5_1", bbox_inches='tight')
