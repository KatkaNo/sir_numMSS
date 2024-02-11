import numpy as np
from skfem import *
from skfem.models.elasticity import *
from skfem.visuals.matplotlib import *
from skfem.helpers import dot

# Problem description: http://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php

L = 10.0
a = 0.015*L
b = 1.0e-5*a
E, nu = 2.5, 0.25 #7.3e4, 0.2
E,nu = plane_stress(E,nu)
Lambda, mu = lame_parameters(E, nu)

coef = 2.22/(4*L**3)
P = coef * 4*E*a**3*b

m = MeshQuad.init_tensor(np.linspace(0, L, 11), np.linspace(-a, a, 4)).with_boundaries(
    {
        "left": lambda x: x[0] == 0.0,
        "right": lambda x: x[0] == L
    }
)

e = ElementVector(ElementQuadP(1))
gb = Basis(m, e)

K = asm(linear_elasticity(Lambda, mu), gb)

def trac(x, y):
    return np.array([0, -P/(2*a*b)])

@LinearForm
def loadingN(v, w):
    return dot(trac(*w.x), v)

left_basis = FacetBasis(m, e, facets=m.boundaries["left"])

rpN = asm(loadingN, left_basis)

clamped = gb.get_dofs(m.boundaries["right"])

u = solve(*condense(K, rpN, D=clamped)) 

def airy(x1, x2):
    w = 3*coef*L**2
    d = -2*coef*L**3
    return 3*coef*np.multiply(np.power(x1,2),x2) - coef*(2+nu)*np.power(x2,3) + 6*coef*(1+nu)*a*a*x2 - w*x2, \
            -nu*3*coef*np.multiply(x1, np.power(x2,2)) - coef*np.power(x1,3) + w*x1 + d

def visualize():
    import matplotlib.pyplot as plt
    axi = plt.subplot()
    axi.set_aspect(1)

    u_ex = airy(*m.p)
    m1 = m.translated(u[gb.nodal_dofs])
    m2 = m.translated(u_ex)
    
    axi = draw(m1, ax=axi, color='r')
    axi = draw(m2, ax=axi, color='k')
    axi.plot()
    axi.legend(['Finite element', 'Analytical'])
    print("min/max u1:", min(u[gb.nodal_dofs][0]), max(u[gb.nodal_dofs][0]))
    print("min/max u2:", min(u[gb.nodal_dofs][1]), max(u[gb.nodal_dofs][1]))
    print("min/max airy1:", min(u_ex[0]), max(u_ex[0]))
    print("min/max airy2:", min(u_ex[1]), max(u_ex[1]))
    return axi

visualize().show()
