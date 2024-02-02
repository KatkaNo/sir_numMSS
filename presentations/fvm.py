import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import cg

L = 1.0
T = 1.0
N = 8
M = 8

def dirichlet(x,y):
	return np.cos(2*np.pi*x/L)*np.cos(2*np.pi*y/L)

x, y = np.linspace(0,L,N+1), np.linspace(0,L,N+1)
h = L/N

U = np.zeros(shape=(N,N))

A = lil_matrix(( N*N, N*N ))
b = np.zeros(N*N)
# cycle over vertical faces
for j in range(N):
	for i in range(N+1):
		if i == 0:
			g = dirichlet(x[i],y[j])
			r = i*N+j if i == 0 else (i-1)*N+j
			A[r,r] += 1.0/(h/2) * h
			b[r] += g/(h/2) * h
		elif i != N:
			r1, r2 = (i-1)*N+j, i*N+j
			coef = 1.0/h * h
			A[r1,r1] += coef
			A[r1,r2] -= coef
			A[r2,r1] -= coef
			A[r2,r2] += coef
# cycle over horizontal faces
for i in range(N):
	for j in range(N+1):
		if j == 0:
			g = dirichlet(x[i],y[j])
			r = i*N+j if j == 0 else i*N+j-1
			A[r,r] += 1.0/(h/2) * h
			b[r] += g/(h/2) * h
		elif j != N:
			r1, r2 = i*N+j-1, i*N+j
			coef = 1.0/h * h
			A[r1,r1] += coef
			A[r1,r2] -= coef
			A[r2,r1] -= coef
			A[r2,r2] += coef

M = A.tocsr()
sol, exit_code = cg(M, b.flatten(), tol=1.0e-8)
sol = sol.reshape( (N, N) )

Q=M.todense()
plt.imshow(Q,interpolation='none',cmap='binary')
plt.colorbar()
plt.show()
plt.clf()

X, Y = np.meshgrid(0.5*(x[:-1]+x[1:]),0.5*(y[:-1]+y[1:]))
plt.contourf(X,Y,sol)
plt.colorbar()
plt.xlim((0, L))
plt.ylim((0, L))
plt.gca().set_aspect(1)
plt.draw()
plt.show()