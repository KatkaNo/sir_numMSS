import numpy as np
import matplotlib.pyplot as plt

# physical properties
p0 = 0.0 # some reference pressure
phi0, cR = 1.0e0, 1.0e0 # porosity(p0), compressibility
mu0, cMu = 1.0e1, 1.0e0 # viscosity(p0), linear scaling coef
Kx = 1.0 # constant permeability
# analytical solution p(x,t) = (e^{t1(t+t0)x/L}-1)/e^{t1(t+t0)}-1
t0, t1 = 0.1, 10.0
def phi(p):
	# porosity should be between 0 and 1
	return np.maximum(0.0,np.minimum(1.0, phi0 * (1.0 + cR*(p-p0))))
def dphi(p):
	return phi0*cR # dphi/dp
def mu(p):
	return np.maximum(0.0,mu0*(1.0 + cMu*(p-p0)))
def dmu(p):
	return mu0*cMu
def p_exact(x,t):
	t += t0
	t *= t1
	return (np.exp(t*x/L)-1.)/(np.exp(t)-1)
def dpdt_exact(x,t):
	t += t0
	t *= t1
	return (np.exp(t*x/L)*( (x/L-1)*np.exp(t) - x/L) + np.exp(t))/(np.exp(t)-1)**2
def dpdx_exact(x,t):
	t += t0
	t *= t1
	return t*np.exp(t*x/L)/(np.exp(t)-1)
def d2pdx2_exact(x,t):
	t += t0
	t *= t1
	return t*t*np.exp(t*x/L)/(np.exp(t)-1)
def phi_exact(x,t):
	return phi(p_exact(x,t))
def mu_exact(x,t):
	return mu(p_exact(x,t))
def f_exact(x,t):
	return phi_exact(x,t)*dpdt_exact(x,t) - Kx*(d2pdx2_exact(x,t)-(cMu*mu0)*np.power(dpdx_exact(x,t),2)/mu_exact(x,t))/mu_exact(x,t)

def lmbd(p):
	return Kx/mu(p)
def harmavg(a1, a2):
	return 2*a1*a2/(a1+a2+1.0e-20)
def upwind(i,P, func=lmbd):
	p1, p2 = P[i], P[i+1]
	return func(p1) if p1 > p2 else func(p2)
def iupw(i,P):
	return i if P[i] > P[i+1] else i+1

def solve_progonka(a,D,c,rhs):
    d = rhs.copy()
    b = D.copy()
    size = d.size
    for i in range(size-1):
        w = a[i]/b[i]
        b[i+1] = b[i+1] - w*c[i]
        d[i+1] = d[i+1] - w*d[i]
    x = np.empty_like(d)
    x[size-1] = d[size-1]/b[size-1]
    for i in range(size-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1]) / b[i]
    return x

def matvec(LD,D,UD,V):
    size = D.size
    y = np.empty_like(V)
    for i in range(size):
        ax_i = D[i] * V[i]
        ax_i = ax_i + (0.0 if i == 0 else LD[i-1] * V[i-1])
        ax_i = ax_i + (0.0 if i == size-1 else UD[i] * V[i+1])
        y[i] = ax_i
    return y

def q_left(t):
	return 1.0#t**0.5*np.power(np.sin(np.pi/2*t), 2)

def assemble_tridiag_linearization(P0,h,dt, t):
	D, F = np.zeros(P0.size), np.zeros(P0.size)
	LD, UD = np.zeros(P0.size-1), np.zeros(P0.size-1)
	for i in range(P0.size):
		if i == 0:
			# left boundary: Neumann boundary condition -kx*lw*dP/dX = q_left
			# T = -Kx*lmbd(P0[0])/h
			# D[0], UD[0] = -T, T
			# F[0] = q_left(t)
			# left boundary: Dirichlet p = p0
			D[0], UD[0], F[0] = 1.0, 0.0, P0[0]
		# right boundary: p = p_right
		elif i == x.size - 1:
			LD[-1], D[-1], F[-1] = 0.0, 1.0, P0[-1]
		else:
			gamma = dt/(h*h)
			lmbd_left, lmbd_right = upwind(i-1,P0), upwind(i,P0)
			Tm, Tp = gamma*lmbd_left, gamma*lmbd_right
			LD[i-1], D[i], UD[i] = -Tm, Tm+Tp, -Tp
			D[i] += phi(P0[i])
			F[i] = phi(P0[i])*P0[i]
			F[i] += f_exact(x[i],t)*dt
	return LD,D,UD,F

def assemble_tridiag_newton(P0,P1,h,dt, t):
	P = P0
	D, F = np.zeros(P.size), np.zeros(P.size)
	LD, UD = np.zeros(P.size-1), np.zeros(P.size-1)
	for i in range(P.size):
		if i == 0:
			# left boundary: Neumann boundary condition -kx*lw*dP/dX = q_left
			# T = -Kx*lmbd(P[0])/h
			# D[0], UD[0] = -T, T
			# F[0] = T*(P[1]-P[0]) - q_left(t)
			# left boundary: Dirichlet p = p0
			D[0], UD[0], F[0] = 1.0, 0.0, 0.0
		# right boundary: p = p_right
		elif i == x.size - 1:
			LD[-1], D[-1], F[-1] = 0.0, 1.0, 0.0
		else:
			gamma = dt/(h*h)
			lmbd_left, lmbd_right = upwind(i-1,P), upwind(i,P)
			Tm, Tp = gamma*lmbd_left, gamma*lmbd_right
			i_left, i_right = iupw(i-1,P), iupw(i,P)
			mult_left = (1.0-dmu(P[i_left])/mu(P[i_left])*P[i_left])
			mult_right = (1.0-dmu(P[i_right])/mu(P[i_right])*P[i_right])
			LD[i-1] = -Tm*(mult_left if i_left == i-1 else 1.0)
			UD[i] = -Tp*(mult_right if i_right == i+1 else 1.0)
			D[i] = Tm*(mult_left if i_left == i else 1.0) + Tp*(mult_right if i_right == i else 1.0)
			D[i] += phi(P[i]) + dphi(P[i])*P[i]
			F[i] = phi(P[i])*(P0[i]-P1[i]) - gamma*( lmbd_right*(P[i+1]-P[i]) - lmbd_left*(P[i]-P[i-1]) )
			F[i] -= f_exact(x[i],t)*dt
	return LD,D,UD, F

L = 1.0 # reservoir length
T = 1.0 # final time
N = 100 # divide length by N
M = 200 # amount of time steps
Nsave = 20
p_left, p_right = 0.0, 1.0 # pressure at right boundary
#q_left = 1.0 # flux from left boundary 
x = np.linspace(0., L, N+1)
h, dt = L/M, T/N # space/time step
# initial conditions: P = 0
P = np.zeros(shape=(x.size, M+1))
P[:,0] = p_exact(x,0.0)
# setup boundary conditions
P[0,:] = p_left # fixed pressure on left boundary
P[-1,:] = p_right # fixed pressure on right boundary

frames = 10
t_save = T/frames 

def should_save_frame(t,dt):
	return abs(t-(int(t/t_save)+1)*t_save) < dt

def linearization():
	print(f'Linearization: X steps {N} T steps {M}')
	# time loop
	i, t = 0, 0.0
	while t < T+1.0e-10:
		P0 = np.copy(P[:,i])
		LD,D,UD,F = assemble_tridiag_linearization(P0,h,dt, t)
		P[:,i+1] = solve_progonka(LD,D,UD,F)
		if should_save_frame(t,dt):
			plt.plot(x, P[:,i+1],label=f't={t:.01f}')
			plt.xlim(0,L)
			plt.draw()
		i += 1
		t += dt
	plt.legend()
	plt.savefig('out_linear.png')

def newton():
	print(f'Newton method: X steps {N} T steps {M}')
	# time loop
	i, t = 0, 0.0
	while t < T+1.0e-10:
		P1 = np.copy(P[:,i])
		P0 = np.copy(P1)
		iters, maxiters = 0, 10000
		# absolute, relative and divergence tolerances
		atol, rtol, divtol = 1.0e-10, 1.0e-4, 1.0e10
		converged, fail = False, False
		while not (converged or fail):
			LD,D,UD, F = assemble_tridiag_newton(P0,P1,h,dt, (t+dt))
			res = np.linalg.norm(F,ord=2)
			if iters == 0:
				res0 = res
			if res < atol or res < rtol * rtol:
				converged = True
			elif res > divtol or iters >= maxiters:
				fail = True
			P0 += solve_progonka(LD,D,UD,-1.*F)
			#input(f"iter {iters} res {res} res0 {res0}")
			iters += 1
		if fail:
			input(f"i {i} iters {iters} res {res} res0 {res0}")
		P[:,i+1] = P0
		if should_save_frame(t,dt):
			# plt.plot(x, p_exact(x,t),c='r',label=f't={t:.01f}')
			plt.plot(x, P[:,i+1],label=f't={t:.01f}')
			# plt.plot(x, p_exact(x,t)-P[:,i+1],c='b',label=f't={t:.01f}')
			plt.xlim(0,L)
			plt.draw()
		i += 1
		t += dt
	plt.legend()
	plt.savefig('out_newton.png')

def explicit():
	print(f'Explicit method: X steps {N} T steps {M}')
	# time loop
	i, t = 0, 0.0
	while t < T+1.0e-10:
		P0 = np.copy(P[:,i])
		P1 = np.copy(P0)
		F = np.array([ 1.0/phi(P1[j])*( upwind(j,P1)*(P1[j+1]-P1[j]) - upwind(j-1,P1)*(P1[j]-P1[j-1]) ) \
			for j in range(1,P0.size-1)])
		P0[1:-1] = P1[1:-1] + dt/(h*h)*F + dt*f_exact(x[1:-1],t)/phi(P1[1:-1])
		# given flux on left boundary:
		# TM = -Kx*lmbd(P1[0])/h
		# P0[0] = P0[1] - q_left(t/T)/TM
		# given pressure on left boundary:
		P[:,i+1] = P0
		assert not np.any(np.isnan(P0)) and not np.any(np.isinf(P0)), 'NaN or inf found'
		if should_save_frame(t,dt):
			plt.plot(x, P[:,i+1],label=f't={t:.01f}')
			plt.xlim(0,L)
			plt.draw()
		i += 1
		t += dt
	plt.legend()
	plt.savefig('out_expl.png')

if __name__ == "__main__":
	# linearization()
	newton()
	# explicit()