from dolfin import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Dirichlet boundary (x=0 or y=0 or y=1)
def boundary(x):
	return x[0] < DOLFIN_EPS or \
		x[1] < DOLFIN_EPS or \
		x[1] > 1 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational formulation of
# f(x,y)=2x, g(y)=y(1-y)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("2.0*x[0]", degree=2)
g = Expression("x[1]*(1-x[1])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Define ue to be the exact solution, xy(1-y)
# and create a pw linear interpolant of it
ue = Expression("x[0]*x[1]*(1-x[1])", degree=2)
uze = interpolate(ue,V)

# For convenience, define the difference
diff = u-uze

# Compute norms
h1_err = errornorm(u, uze, norm_type='H1', degree_rise=3)

# Plot solutions, difference, and norm
fig=plt.figure(figsize=(16,4)) 
plt.suptitle('Computed solution (u) vs piecewise linear \
		interpolant of exact solution (uze):  H1 error='\
		+"%.2e"%h1_err, size='xx-large', y=1.1)

plt.subplot(131, title = 'u')
plt.colorbar(plot(u, interactive=True))

plt.subplot(132, title = 'uze')
plt.colorbar(plot(uze, interactive=True))

plt.subplot(133, title = 'u-uze')
plt.set_cmap('spring')
plt.colorbar(plot(diff, interactive=True))

plt.show()

# Save plot as png
fig.savefig("figure2_6", bbox_inches='tight')
