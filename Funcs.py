from fenics import *
import numpy as np
import ufl as uf
import meshio

def magnitude(u):

    return sqrt(u**2)

def epsilon(u): 

    return sym(grad(u))

def trans(u):

    return u.T

def mgrad(phi):

    return sqrt(dot(grad(phi),grad(phi)))

def new_norm(phi):

    return grad(phi) / mgrad(phi)

def delta_func(phi, eps):

    return conditional(lt(abs(phi),eps), 1.0/(2.0*eps)*(1.0 + uf.cos(np.pi*phi/eps)), 0.0)
    
def heaviside(phi, eps):

    return conditional(lt(abs(phi),eps), 0.5*(1.0 + phi/eps + 1/np.pi*uf.sin(np.pi*phi/eps)), (uf.sign(phi) + 1)/2.0)

def secondinvariant(u):

   return pow(0.5*inner(grad(u)+grad(u).T,grad(u)+grad(u).T),0.5)

def powerlaw(u, n):

    return secondinvariant(u)**(n-1)

def carreau(u, n, eta_0, eta_inf, lamb):

    return eta_inf+(eta_0-eta_inf)*(1+(lamb*secondinvariant(u))**2)**(0.5*(n-1))

def rho_noncon(phi, rho1, rho2, eps):

    return rho2*heaviside(phi, eps) + rho1*(1-heaviside(phi, eps))

def mu_noncon(phi, mu1, mu2, eps):
    
    return mu2*heaviside(phi, eps) + mu1*(1-heaviside(phi, eps))

def rho_con(phi, rho1, rho2):

    return rho1*phi + rho2*(1-phi)

def mu_con(phi, mu1, mu2):
    
    return mu1*phi + mu2*(1-phi)

def sgn(phi, eps1):
    
    return phi/(sqrt(phi**2+eps1**2*(dot(grad(phi),grad(phi)))))

def ngamma(phi):

    return grad(phi)/mgrad(phi)

def eta_s_con(phi, eta_s_in, eta_s_out):

    return eta_s_in*phi + eta_s_out*(1-phi)

def eta_p_con(phi, eta_p_in, eta_p_out):

    return eta_p_in*phi + eta_p_out*(1-phi)

def lamb1_con(phi, lamb1_in, lamb1_out):

    return lamb1_in*phi + lamb1_out*(1-phi)

def poly_flux(u, n, tau):

    return (dot(u,n)*tau + abs(dot(u,n))*tau)/2

def flux(ui, ni, taui):
    
    return (dot(ui,ni)*taui + abs(dot(ui,ni))*taui)/2

def flux1(ui, ni):
    return (dot(ui, ni) + abs(dot(ui, ni)))/2

def phasify_con(phi, param_in, param_out):

    return param_in*phi + param_out*(1-phi)

def phasify_noncon(phi, param_in, param_out, eps):
    
    return param_out*heaviside(phi, eps) + param_in*(1-heaviside(phi, eps))

def create_mesh(mesh, cell_type, prune_z=False):

    cells = mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells})

    if prune_z:
        out_mesh.prune_z_0()

    return out_mesh

# def grad_axi(u,x):

#     return as_tensor(as_tensor([[u[0].dx(0), 0, u[0].dx(1)],
#                           [0, u[0] / x[0], 0],
#                           [u[1].dx(0), 0, u[1].dx(1)]]))

def nabla_grad_axi(u,x):

    return as_tensor([[u[0].dx(0), 0, u[1].dx(0)],
                          [0, u[0] / x[0], 0],
                          [u[0].dx(1), 0, u[1].dx(1)]])

def gradp_axi(p,x):

    return as_vector([p.dx(0),p.dx(1)])

def epsilon_axi(u,x):

    return sym(as_tensor([[u[0].dx(0), 0, u[0].dx(1)],
                          [0, u[0] / x[0], 0],
                          [u[1].dx(0), 0, u[1].dx(1)]]))

def div_axi(u, x):

    return (1/x[0])*(x[0]*u[0]).dx(0) + u[1].dx(1)

def mgrad_axi(phi, x):

    return sqrt(dot(gradp_axi(phi,x),gradp_axi(phi,x)))

def ngamma_axi(phi,x):

    return gradp_axi(phi,x)/mgrad_axi(phi,x)

# def dim_to_nondim(fluid, 
#                   length_dim,
#                   height_dim,
#                   grav,
#                   surf_tens,
#                   T_dim,
#                   dt_dim,
#                   size):

#     mesh_nondim = RectangleMesh(Point(0,0),Point(length_dim, height_dim),size,2*size) 

#     if (fluid == 'Newtonian'):
        
#         rho_in = 
#         rho_out = 
#         Re = 
#         Fr = 
#         We = 

#     if (fluid == 'Viscoelastic'):

#         pass

#     Re = 



#     return pass

# def nondim_to_dim():

#     return 



"""good results were eps: 0.9 dtau: 1.10 size: 200 euler scheme old method order 2"""

# elif (self.fluid == 'GNF_PL'):

#     self.mu = phasify_con(self.phi, self.mu1[0], powerlaw(self.u0, self.pli_out))

# elif (self.fluid == 'GNF_C'):

#     self.mu = phasify_con(self.phi, 
#                     carreau(self.u0, self.pli_in, self.eta0_in, self.etainf_in, self.lamb_in), 
#                     carreau(self.u0, self.pli_out, self.eta0_out, self.etainf_out, self.lamb_out))

# follows
# follows
# false
# for b d