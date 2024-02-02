"""
Solve poisson problem:
-d2p/dx2 = f, x in (0,L)
p(0) = p(L) = 0
with mixed finite element method 
"""

import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

def p_exact(x):
	return np.sin(2*np.pi*x)
# u = -dp/dx
def u_exact(x):
	return -2*np.pi*np.cos(2*np.pi*x)
# f = -d2p/dx2
def f(x):
	return 4*np.pi**2*np.sin(2*np.pi*x)
# \int\limits_{x_0}^{x_1} f(x) dx = - (dp/dx(x_1) - dp/dx(x_0))
def int_f(x0, x1):
	return -(u_exact(x1) - u_exact(x0))

# M -- total amount of nodes including boundaries
def run(M):
	L = 1.0
	h = L/(M-1)
	x = np.linspace(0,L,M) # nodes
	xmid = 0.5*(x[1:]+x[:-1]) # midpoints between nodes

	p_left, p_right = p_exact(0.0), p_exact(L) # Dirichlet boundary condition

	dA = np.ones(M)*4
	dA[0] = dA[-1] = 2
	lA = np.ones(M-1)
	A = h/6*(np.diag(dA,0) + np.diag(lA,1) + np.diag(lA,-1))

	B = np.zeros(shape=(M,M-1))
	iB,jB = np.indices(B.shape)
	B[iB==jB] = 1.
	B[iB==jB+1] = -1.

	O = np.zeros(shape=(M-1,M-1))

	K = np.block(
		[[A, B],
		 [B.T,O]]
	)

	fU = np.zeros(M)
	fU[0] = p_left
	fU[-1] = -p_right
	fP = -int_f(x[1:], x[:-1])

	f = np.hstack([
		fU,
		fP
		])
	u = solve(K,f)

	# plt.plot(xmid, u[M:], 'b-')
	# plt.plot(xmid, p_exact(xmid), 'r-')
	# plt.plot(x, u[:M], 'b.')
	# plt.plot(x, u_exact(x), 'r-')
	# plt.show()

	def l2err(du):
		return np.sqrt(h*np.dot(du,du))
	print(f'L2_error: du/dx {l2err(u[:M]-u_exact(x))} u {l2err(u[M:]-p_exact(xmid))}')

if __name__ == "__main__":
	run(11)
	run(101)
	run(1001)