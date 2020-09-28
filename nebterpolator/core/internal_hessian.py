"""
Numpy routines to calculate the analytical second derivatibes of internal
coordinates (bonds, angles, dihedrals) with respect to cartesian coordinates
"""

##############################################################################
# Imports
##############################################################################
from __future__ import print_function
import numpy as np

from . import internal
from . import internal_derivs

##############################################################################
# GLOBALS
##############################################################################

VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)

__all__ = ['bond_hessian', 'angle_heddian', 'dihedral_hessian']

##############################################################################
# Functions
##############################################################################


def sign3(i, j, k):
    if i == j:
        return 1
    if i == k:
        return - 1
    else:
        return 0


def sign6(a, b, c, i, j, k):
    return sign3(a,b,c)*sign3(i,j,k)


def bond_hessian(xyz, ibonds):
    """
    Hessian of the bond lengths with respect to cartesian coordinates

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation.
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    bond_hessian : np.ndarray, shape=[n_bonds, n_atoms, 3, n_atoms, 3]
        The hessian is a 5d array, where bond_derivs[i,l,m,n,o] gives the
        derivative of the `i`th bond length (the bond between atoms
        ibonds[i,0] and ibonds[i,1]) with respect to the `l`th atom's
        `m`th coordinate and the `n`th atom's `o`th coordinate.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_bonds, two = ibonds.shape
    if two != 2:
        raise TypeError('ibonds must have 2 columns.')
    
    hessian = np.zeros((n_bonds, n_atoms, 3, n_atoms, 3))
    for b, (m, n) in enumerate(ibonds):
        u_prime = (xyz[m] - xyz[n])
        length = np.linalg.norm(u_prime)
        u = u_prime / length
        term = (np.outer(u, u) - np.eye(3)) / length

        hessian[b, m, :, n, :] = term
        hessian[b, m, :, m, :] = -term
        hessian[b, n, :, m, :] = term
        hessian[b, n, :, n, :] = -term
    
    return hessian


def angle_hessian(xyz, iangles):
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_angles, three = iangles.shape
    if three != 3:
        raise TypeError('angles must have 3 columns.')
    
    qa = internal.angles(xyz.reshape(1, n_atoms, 3), iangles).flatten()
    jacobian = internal_derivs.angle_derivs(xyz, iangles)
    
    hessian = np.zeros((n_angles, n_atoms, 3, n_atoms, 3))
    for i, (m, o, n) in enumerate(iangles):
        u_prime = xyz[m] - xyz[o]
        v_prime = xyz[n] - xyz[o]
        lambda_u = np.linalg.norm(u_prime)
        lambda_v = np.linalg.norm(v_prime)
        u = u_prime / lambda_u
        v = v_prime / lambda_v
        jac = jacobian[i]

        cos = np.cos(qa[i])
        sin = np.sin(qa[i])
        uv = np.outer(u, v)
        uu = np.outer(u, u)
        vv = np.outer(v, v)
        eye = np.eye(3)
        
        term1 = (uv + uv.T + (-3 * uu + eye) * cos) / (lambda_u**2 * sin)
        term2 = (uv + uv.T + (-3 * vv + eye) * cos) / (lambda_v**2 * sin)
        term3 = (uu + vv - uv   * cos - eye) / (lambda_u * lambda_v * sin)
        term4 = (uu + vv - uv.T * cos - eye) / (lambda_u * lambda_v * sin)
        hessian[i] = -(cos / sin) * np.outer(jac.flatten(), jac.flatten()).reshape(n_atoms, 3, n_atoms, 3)
        
        for a in [m, n, o]:
            for b in [m, n, o]:
                sign6(a,m,o, b,m,o)
                hessian[i, a, :, b, :] += sign6(a,m,o, b,m,o) * term1
                hessian[i, a, :, b, :] += sign6(a,n,o, b,n,o) * term2
                hessian[i, a, :, b, :] += sign6(a,m,o, b,n,o) * term3
                hessian[i, a, :, b, :] += sign6(a,n,o, b,m,o) * term4
    
    return hessian


def dihedral_hessian(xyz, idihedrals):
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_dihedrals, four = idihedrals.shape
    if four != 4:
        raise TypeError('idihedrals must have 4 columns.')
        
    def sym_outer(d1, d2):
        o = np.outer(d1, d2)
        return o + o.T
    
    jacobian = np.zeros((n_dihedrals, n_atoms, 3))
    hessian = np.zeros((n_dihedrals, n_atoms, 3, n_atoms, 3))

    for d, (m, o, p, n) in enumerate(idihedrals):
        u_prime = (xyz[m] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        w_prime = (xyz[p] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        cross_uw = np.cross(u, w)
        cross_vw = np.cross(v, w)
        cos_u = np.dot(u, w)
        cos_v = -np.dot(v, w)
        
        sin2_u = (1 - cos_u**2)
        sin2_v = (1 - cos_v**2)
        sin4_u = sin2_u**2
        sin4_v = sin2_v**2
        
        # JACOBIAN
        jac_term1 = cross_uw / (u_norm * sin2_u)
        jac_term2 = cross_vw / (v_norm * sin2_v)
        jac_term3 = cross_uw * cos_u / (w_norm * sin2_u)
        jac_term4 = cross_vw * cos_v / (w_norm * sin2_v)
        
        for a in [m, o, p, n]:
            jacobian[d, a, :] += sign3(a, m, o)*jac_term1
            jacobian[d, a, :] += sign3(a, p, n)*jac_term2
            jacobian[d, a, :] += sign3(a, o, p)*(jac_term3 - jac_term4)
        # END JACOBIAN
        
        term1 = sym_outer(cross_uw, w*cos_u - u) / (u_norm**2 * sin4_u)

        # The plus sign here in w*cos_v + v is correct, the 2002 paper is wrong.
        term2 = sym_outer(cross_vw, w*cos_v + v) / (v_norm**2 * sin4_v)

        term3 = sym_outer(cross_uw, w - 2*u*cos_u + w*cos_u**2) / (
                    2 * u_norm * w_norm * sin4_u)

        term4 = sym_outer(cross_vw, w + 2*v*cos_v + w*cos_v**2) / (
                    2 * v_norm * w_norm * sin4_v)

        term5 = sym_outer(cross_uw, u + u*cos_u**2 - 3*w*cos_u + w*cos_u**3) / (
                    2 * w_norm**2 * sin4_u)
        term6 = sym_outer(cross_vw, v + v*cos_v**2 + 3*w*cos_v - w*cos_v**3) / (
                    2 * w_norm**2 * sin4_v)    

        term7 = (-w*cos_u + u) / (u_norm * w_norm * (1-cos_u**2))
        term8 = (-w*cos_v - v) / (v_norm * w_norm * (1-cos_v**2))
        
        for a in [m, o, p, n]:
            for b in [m, o, p, n]:
                hessian[d,a,:,b,:] += sign6(a,m,o, b,m,o) * term1
                hessian[d,a,:,b,:] += sign6(a,n,p, b,n,p) * term2
                hessian[d,a,:,b,:] += (sign6(a,m,o, b,o,p) + sign6(a,p,o, b,o,m)) * term3
                hessian[d,a,:,b,:] += (sign6(a,n,p, b,p,o) + sign6(a,p,o, b,n,p)) * term4
                hessian[d,a,:,b,:] += sign6(a,o,p, b,p,o) * term5
                hessian[d,a,:,b,:] += sign6(a,o,p, b,o,p) * term6
                if (a, b) not in [(o,m), (p,m), (n,m), (p,o), (n,o), (n,p)]: continue
                _term7 = sign6(a,p,o, b,o,m)*term7
                _term8 = sign6(a,n,p, b,p,o)*term8
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if i != j and i != k and j != k:
                                hessian[d, a, i, b, j] += (j-i)*(-1.0/2.0)**np.abs(j-i)*(_term7[k] + _term8[k])
                                hessian[d, b, j, a, i] += (j-i)*(-1.0/2.0)**np.abs(j-i)*(_term7[k] + _term8[k])
    return jacobian, hessian
    
        
if __name__ == '__main__':
    from . import internal_derivs
    #np.random.seed(10)
    
    h = 1e-10
    xyz = np.random.randn(4,3)
    xyz2 = xyz.copy()
    xyz2[1,1] += h
    ibonds = np.array([[0,1], [0,2]])
    iangles = np.array([[0,1,2], [1,2,3]])
    idihedrals = np.array([[0,1,2,3]])

    # print 'TESTING BOND HESSIAN'
    # jac1 = internal_derivs.bond_derivs(xyz, ibonds)
    # jac2 = internal_derivs.bond_derivs(xyz2, ibonds)
    # hessian = bond_hessian(xyz, ibonds)
    # print ((jac2-jac1)/h)[0]
    # print hessian[0, 1, 1]
    
    
    print('\nTESTING ANGLE HESSIAN')
    jac1 = internal_derivs.angle_derivs(xyz, iangles)
    jac2 = internal_derivs.angle_derivs(xyz2, iangles)
    hessian = angle_hessian(xyz, iangles)
    print(((jac2-jac1)/h)[0])
    print()
    print(hessian[0, 1, 1]) 
    
    print('\nTESTING DIHEDRAL HESSIAN')
    jac1, hessian = dihedral_hessian(xyz, idihedrals)
    jac2, hessian = dihedral_hessian(xyz2, idihedrals)

    print('These matricies should match')
    print(((jac2-jac1)/h)[0])
    print() 
    print(hessian[0][1,1])
    
