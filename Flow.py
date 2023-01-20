from fenics import *
import ufl as uf
from Funcs import *

class Flow():

    """Choose linear solvers for solving equations"""
    def __init__(self):
        
        self.ls_linear_solver = 'gmres'
        self.ls_preconditioner = 'default'
        self.lsr_linear_solver = 'gmres'
        self.lsr_preconditioner = 'default'
        self.ns_linear_solver = 'gmres'
        self.ns_preconditioner = 'default'
        self.constitutive_equation_linear_solver = 'bicgstab'
        self.constitutive_equation_preconditioner = 'default'

        if (self.element == 'DG'):

            parameters['ghost_mode'] = 'shared_facet' 

        # parameters["form_compiler"]["quadrature_degree"] = 5 
        self.Id = Identity(int(self.dimension[0]))
        
    """Construct weak form for level set equation"""
    def ls_form(self):

        if (self.ls_scheme == 'Euler'):

            if (self.element == 'CG'):

                F = (self.phi/self.dt)*self.psi*dx \
                - (self.phi0/self.dt)*self.psi*dx \
                + inner(self.u0, grad(self.phi))*self.psi*dx \

            elif (self.element == 'DG'):

                un = abs(dot(self.u0('+'), self.fn('+')))

                F = (1/self.dt)*(self.phi-self.phi0)*self.psi*dx \
                  - dot ( grad ( self.psi ), self.u0 * self.phi ) * dx \
                  - div(self.u0)*self.phi*self.psi*dx \
                  + (dot(self.u0('+'), jump(self.psi, self.fn))*avg(self.phi) \
                  + 0.5*un*dot(jump(self.phi, self.fn), jump(self.psi, self.fn)))*dS \

        elif (self.ls_scheme == 'CN'):

            F = (self.phi/self.dt)*self.psi*dx \
              - (self.phi0/self.dt)*self.psi*dx \
              + 0.5*inner(self.u0, grad(self.phi))*self.psi*dx \
              + 0.5*inner(self.u0, grad(self.phi0))*self.psi*dx \

        elif (self.ls_scheme == 'CN_SUPG'):

            scaling = 0.1
            toler = 1.0e-3

            if (self.axisymmetry == 'Axi'):

                F = (self.phi/self.dt)*self.psi*self.x_axi[0]*dx \
                - (self.phi0/self.dt)*self.psi*self.x_axi[0]*dx \
                + inner(self.u0, grad(self.phi))*self.psi*self.x_axi[0]*dx \
                + ((self.phi - self.phi0)/self.dt \
                + inner(self.u0, grad(self.phi))) \
                * scaling*self.hmin \
                / uf.Max(2.0*sqrt(inner(self.u0,self.u0)),toler/self.hmin) \
                * inner(self.u0, grad(self.psi))*self.x_axi[0]*dx

            elif (self.axisymmetry == 'No'):
            
                F = (self.phi/self.dt)*self.psi*dx \
                - (self.phi0/self.dt)*self.psi*dx \
                + inner(self.u0, grad(self.phi))*self.psi*dx \
                + ((self.phi - self.phi0)/self.dt \
                + inner(self.u0, grad(self.phi))) \
                * scaling*self.hmin \
                / uf.Max(2.0*sqrt(inner(self.u0,self.u0)),toler/self.hmin) \
                * inner(self.u0, grad(self.psi))*dx
        
        self.a_ls = lhs(F)
        self.m_ls = rhs(F)

        self.A_ls = PETScMatrix()
        self.M_ls = PETScVector()

    def n_form(self):

        if (self.axisymmetry == 'Axi'):

            F = inner((self.phin - ngamma(self.phi0)), self.psin)*self.x_axi[0]*dx

        elif (self.axisymmetry == 'No'):

            F = inner((self.phin - ngamma(self.phi0)), self.psin)*dx

        self.F_nf = F
        # self.a_nf = lhs(F)
        # self.m_nf = rhs(F)

        # self.A_nf = PETScMatrix()
        # self.M_nf = PETScVector()

    """Construct weak form for non-conservative level set reinitialisation"""
    def nclsr_form(self):

        F = (self.phi_rein/self.dtau)*self.psi*dx - (self.phi0 /self.dtau)*self.psi*dx \
          - self.sign*(1-sqrt(dot(grad(self.phi0),grad(self.phi0))))*self.psi*dx \
          + self.alph*inner(grad(self.phi0),grad(self.psi))*dx

        self.F_nclsr = F

        # self.a_ncf = lhs(F)
        # self.m_ncf = rhs(F)

        # self.A_ncf = PETScMatrix()
        # self.M_ncf = PETScVector()

    """Construct weak form for conservative level set reinitialisation"""
    def clsr_form(self):

        if (self.axisymmetry == 'Axi'):

            F = (self.phi_rein - self.phi0)/self.dtau*self.psi_rein*self.x_axi[0]*dx

        elif (self.axisymmetry == 'No'):

            F = (self.phi_rein - self.phi0)/self.dtau*self.psi_rein*dx

        # if (self.element == 'CG'):

        if (self.lsr_method == 'old_method'):

            if (self.axisymmetry == 'Axi'):

                terms = - self.compressor*self.phi_rein*(1.0 - self.phi_rein)*inner(grad(self.psi_rein), self.phin)*self.x_axi[0]*dx \
                        +  self.eps*inner(grad(self.phi_rein), grad(self.psi_rein))*self.x_axi[0]*dx  
            
            elif (self.axisymmetry == 'No'):

                terms = -  self.compressor*self.phi_rein*(1.0 - self.phi_rein)*inner(grad(self.psi_rein), self.phin)*dx \
                        +  self.eps*inner(grad(self.phi_rein), grad(self.psi_rein))*dx  

            # terms = - 0.5*dot(grad(self.psi_rein),(self.phi_rein+self.phi0)*self.phin)*dx \
            #         + dot(self.phi_rein*self.phi0*self.phin,grad(self.psi_rein))*dx \
            #         + (self.eps/2)*(dot(grad(self.phi_rein)+grad(self.phi0),grad(self.psi_rein)))*dx

        elif (self.lsr_method == 'new_method'):
            
            terms = - 0.5*dot(grad(self.psi_rein),(self.phi_rein+self.phi0)*self.phin)*dx \
                    + dot(self.phi_rein*self.phi0*self.phin,grad(self.psi_rein))*dx \
                    + (self.eps/2)*(dot(grad(self.phi_rein)+grad(self.phi0),self.phin)*dot(grad(self.psi),self.phin))*dx
    
        F += terms

        # elif (self.element == 'DG'):
          
        #     phin = self.phi_grad / sqrt(inner(self.phi_grad,self.phi_grad))

        #     nn0 = abs(dot(phin('+'), self.fn('+')))
        #     self.penalty = 100

        #     compressive_terms = - dot ( grad ( self.psi ), phin * self.phi_rein*(1-self.phi_rein) ) * dx \
        #                       + (dot(phin('+'), jump(self.psi, self.fn))*avg(self.phi_rein) \
        #                       + 0.5*nn0*dot(jump(self.phi_rein, self.fn), jump(self.psi, self.fn)))*dS \
        #                       - (dot(phin('+'), jump(self.psi, self.fn))*avg(self.phi_rein*self.phi_rein) \
        #                       + 0.5*nn0*dot(jump(self.phi_rein*self.phi_rein, self.fn), jump(self.psi, self.fn)))*dS \

        #     diffusive_terms = self.eps*(inner(grad(self.phi_rein), grad(self.psi))*dx \
        #                     + (self.penalty/self.hmin)*dot(jump(self.psi, self.fn), jump(self.phi_rein, self.fn))*dS \
        #                     - dot(avg(grad(self.psi)), jump(self.phi_rein, self.fn))*dS \
        #                     - dot(jump(self.psi, self.fn), avg(grad(self.phi_rein)))*dS)

        #     F += compressive_terms
        #     F += diffusive_terms

        self.F_clsr = F

        # self.a_cf = lhs(F)
        # self.m_cf = rhs(F)

        # self.A_cf = PETScMatrix()
        # self.M_cf = PETScVector()

    """Construct the weak form for the DEVSSG velocity gradient tensor projection"""
    def DEVSSG_form(self):

        F = inner((self.G - grad(self.u0)),self.R)*dx
        F_tt = inner((self.G_tt - self.u0[0]/self.x_axi[0]),self.R_tt)*dx

        self.a_DEVSSG = lhs(F)
        self.m_DEVSSG = rhs(F)

        self.A_DEVSSG = PETScMatrix()
        self.M_DEVSSG = PETScVector()

        self.a_DEVSSG_tt = lhs(F_tt)
        self.m_DEVSSG_tt = rhs(F_tt)

        self.A_DEVSSG_tt = PETScMatrix()
        self.M_DEVSSG_tt = PETScVector()

    """Construct the weak form for the Oldroyd-B viscoelastic constitutive equation."""
    def constitutive_equation_form(self):

        # if (self.DEVSSG == 'True'):

        #     self.GRAD =  self.G1

        #     self.GRAD_tt = self.G1_tt
        
        # else:

        #     self.GRAD = grad(self.u0)

        #     self.GRAD_tt = self.u0[0]/self.x_axi[0]

        # if (self.axisymmetry == 'Axi'):
 
        #     if (self.constitutive_type == 'normal'):

        #         F = 0.5*(inner(self.tau, self.zeta)*self.x_axi[0]*dx + inner(self.tau0, self.zeta)*self.x_axi[0]*dx) \

        #     elif (self.constitutive_type == 'conf'):

        #         F = 0.5*(inner(self.tau-self.Id, self.zeta)*self.x_axi[0]*dx+inner(self.tau0-self.Id, self.zeta)*self.x_axi[0]*dx) \

        # elif (self.axisymmetry == 'No'):

        #     if (self.constitutive_type == 'normal'):

        #         F = 0.5*(inner(self.tau, self.zeta)*dx + inner(self.tau0, self.zeta)*dx) \

        #     elif (self.constitutive_type == 'conf'):

        #         F = 0.5*(inner(self.tau-self.Id, self.zeta)*dx+inner(self.tau0-self.Id, self.zeta)*dx) \

        if (self.dimensional == 'Dim'): 

            if (self.axisymmetry == 'No'):

                F += (1/self.dt)*inner(self.lamb*(self.tau-self.tau0),self.zeta)*dx \
                    + 0.5*(inner(dot(self.lamb*self.u0,nabla_grad(self.tau)) \
                    - dot(self.lamb*self.tau, trans(self.GRAD)) \
                    - dot(self.lamb*self.GRAD, self.tau),self.zeta)*dx) \
                    + 0.5*(inner(dot(self.lamb*self.u0,nabla_grad(self.tau0)) \
                    - dot(self.lamb*self.tau0, trans(self.GRAD)) \
                    - dot(self.lamb*self.GRAD, self.tau0),self.zeta)*dx) \

                if (self.ps_element == 'DG'):

                    F += self.lamb*0.5*inner((flux(self.u0,self.fn,self.tau)('+') - flux(self.u0,self.fn,self.tau)('-')),jump(self.zeta))*dS \
                        + self.lamb*0.5*inner((flux(self.u0,self.fn,self.tau0)('+') - flux(self.u0,self.fn,self.tau0)('-')),jump(self.zeta))*dS

                if (self.constitutive_type == 'normal'):

                    F += - self.eta_p*inner(self.GRAD+self.GRAD.T,self.zeta)*dx \
            
                if (self.constitutive_equation == 'Giesekus'):

                    F += (self.gmf*self.lamb/self.eta_p)*inner(self.tau0*self.tau0,self.zeta)*dx \

            elif (self.axisymmetry == 'Axi'):

                if (self.DEVSSG != 'True'):

                    self.G1 = grad(self.u0)
                    self.G1_tt = self.u0[0]/self.x_axi[0]

                F = (1/self.dt)*inner(self.lamb*(self.tau-self.tau0),self.zeta)*self.x_axi[0]*dx\
                    + 0.5*(inner(dot(self.lamb*self.u0,nabla_grad(self.tau)),self.zeta)*self.x_axi[0]*dx \
                    - inner(dot(self.lamb*self.tau, trans(self.G1)) \
                    - dot(self.lamb*self.G1, self.tau),self.zeta)*self.x_axi[0]*dx) \
                    + 0.5*(inner(dot(self.lamb*self.u0,nabla_grad(self.tau0)),self.zeta)*self.x_axi[0]*dx\
                    - inner(dot(self.lamb*self.tau0, trans(self.G1)) \
                    - dot(self.lamb*self.G1, self.tau0),self.zeta)*self.x_axi[0]*dx) \

                Ftt = (1/self.dt)*self.lamb*(self.tautt-self.tau0tt)*self.zetatt*self.x_axi[0]*dx \
                    + 0.5*(self.lamb*self.u0[1]*self.tautt.dx(1) + self.lamb*self.u0[0]*self.tautt.dx(0) \
                    - 2*self.lamb*self.tautt*self.G1_tt)*self.zetatt*self.x_axi[0]*dx \
                    + 0.5*(self.lamb*self.u0[1]*self.tau0tt.dx(1) + self.lamb*self.u0[0]*self.tau0tt.dx(0) \
                    - 2*self.lamb*self.tau0tt*self.G1_tt)*self.zetatt*self.x_axi[0]*dx \

                if (self.ps_element == 'DG'):

                    F += self.lamb*0.5*inner((flux(self.u0,self.fn,self.tau)('+') - flux(self.u0,self.fn,self.tau)('-')),jump(self.zeta))*self.x_axi[0]*dS \
                        + self.lamb*0.5*inner((flux(self.u0,self.fn,self.tau0)('+') - flux(self.u0,self.fn,self.tau0)('-')),jump(self.zeta))*self.x_axi[0]*dS

                    Ftt += self.lamb*0.5*inner((flux(self.u0,self.fn,self.tautt)('+') - flux(self.u0,self.fn,self.tautt)('-')),jump(self.zetatt))*self.x_axi[0]*dS \
                        + self.lamb*0.5*inner((flux(self.u0,self.fn,self.tau0tt)('+') - flux(self.u0,self.fn,self.tau0tt)('-')),jump(self.zetatt))*self.x_axi[0]*dS

                if (self.stability == 'SUPG'):

                    h = CellDiameter(self.mesh)
                    supg = (h/magnitude(self.u0)+0.000001)

                    F += 0.5*self.lamb*inner(dot(self.u0, nabla_grad(self.zeta)), supg*dot(self.u0,nabla_grad(self.tau)))*self.x_axi[0]*dx \
                      + 0.5*self.lamb*inner(dot(self.u0, nabla_grad(self.zeta)), supg*dot(self.u0,nabla_grad(self.tau0)))*self.x_axi[0]*dx \

                    Ftt += 0.5*self.lamb*inner((self.u0[1]*self.zetatt.dx(1) + self.u0[0]*self.zetatt.dx(0)), supg*(self.u0[1]*self.tautt.dx(1) + self.u0[0]*self.tautt.dx(0)))*self.x_axi[0]*dx \
                        + 0.5*self.lamb*inner((self.u0[1]*self.zetatt.dx(1) + self.u0[0]*self.zetatt.dx(0)), supg*(self.u0[1]*self.tau0tt.dx(1) + self.u0[0]*self.tau0tt.dx(0)))*self.x_axi[0]*dx \

                if (self.constitutive_type == 'normal'):

                    F += - (self.eta_p)*inner(self.G1+trans(self.G1),self.zeta)*self.x_axi[0]*dx \

                    Ftt += -2*self.eta_p*self.G1_tt*self.zetatt*self.x_axi[0]*dx \
            
                if (self.constitutive_equation == 'Giesekus'):

                    if (self.constitutive_type == 'conf'):

                        F += (self.gmf)*inner((self.tau0-self.Id)*(self.tau0-self.Id),self.zeta)*self.x_axi[0]*dx \

                        Ftt += (self.gmf)*(self.tau0tt-1)*(self.tau0tt-1)*self.zetatt*self.x_axi[0]*dx \

                    elif (self.constitutive_type == 'normal'):

                        eps_ptt = 0.05
                        xi_ptt = 0.12

                        F += 0.5*inner((exp((eps_ptt*self.lamb*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/self.eta_p))*self.tau0,self.zeta)*self.x_axi[0]*dx \
                          + 0.5*inner((exp((eps_ptt*self.lamb*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/self.eta_p))*self.tau,self.zeta)*self.x_axi[0]*dx \
                        
                        # F += 0.5*(xi_ptt*self.lamb*inner((self.G1+trans(self.G1))*self.tau0 + self.tau0*(self.G1+trans(self.G1)),self.zeta)*self.x_axi[0]*dx) \
                        #     + 0.5*(xi_ptt*self.lamb*inner((self.G1+trans(self.G1))*self.tau + self.tau*(self.G1+trans(self.G1)),self.zeta)*self.x_axi[0]*dx) \
                        
                        #(self.gmf*self.lamb/(self.eta_p))*inner(self.tau0*self.tau0,self.zeta)*self.x_axi[0]*dx \

                        Ftt += 0.5*inner((exp((eps_ptt*self.lamb*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/self.eta_p))*self.tau0tt,self.zetatt)*self.x_axi[0]*dx \
                            + 0.5*inner((exp((eps_ptt*self.lamb*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/self.eta_p))*self.tautt,self.zetatt)*self.x_axi[0]*dx \
                        
                        # Ftt += 0.5*(xi_ptt*self.lamb*inner((2*self.u0[0]/self.x_axi[0])*self.tau0tt + self.tau0tt*(2*self.u0[0]/self.x_axi[0]),self.zetatt)*self.x_axi[0]*dx) \
                        #     + 0.5*(xi_ptt*self.lamb*inner((2*self.u0[0]/self.x_axi[0])*self.tautt + self.tautt*(2*self.u0[0]/self.x_axi[0]),self.zetatt)*self.x_axi[0]*dx) \
                        
                        #(self.gmf*self.lamb/(self.eta_p))*self.tau0tt*self.tau0tt*self.zetatt*self.x_axi[0]*dx \
        elif (self.dimensional == 'NonDim'):

            if (self.axisymmetry == 'No'):

                if (self.DEVSSG != 'True'):

                    self.G1 = grad(self.u0)
                    self.G1_tt = self.u0[0]/self.x_axi[0]

                F += (1/self.dt)*inner(self.Wi*(self.tau-self.tau0),self.zeta)*dx \
                    + 0.5*(inner(dot(self.Wi*self.u0,nabla_grad(self.tau)) \
                    - dot(self.Wi*self.tau, trans(self.G1)) \
                    - dot(self.Wi*self.G1, self.tau),self.zeta)*dx) \
                    + 0.5*(inner(dot(self.Wi*self.u0,nabla_grad(self.tau0)) \
                    - dot(self.Wi*self.tau0, trans(self.G1)) \
                    - dot(self.Wi*self.G1, self.tau0),self.zeta)*dx) \

                if (self.ps_element == 'DG'):

                    F += self.Wi*0.5*inner((flux(self.u0,self.fn,self.tau)('+') - flux(self.u0,self.fn,self.tau)('-')),jump(self.zeta))*dS \
                        + self.Wi*0.5*inner((flux(self.u0,self.fn,self.tau0)('+') - flux(self.u0,self.fn,self.tau0)('-')),jump(self.zeta))*dS

                if (self.stability == 'SUPG'):

                    h = 1*CellDiameter(self.mesh)
                    supg = (h/magnitude(self.u0)+0.000001)

                    F += 0.5*self.Wi*inner(dot(self.u0, nabla_grad(self.zeta)), supg*dot(self.u0,nabla_grad(self.tau)))*dx \
                    + 0.5*self.Wi*inner(dot(self.u0, nabla_grad(self.zeta)), supg*dot(self.u0,nabla_grad(self.tau0)))*dx \

                if (self.constitutive_type == 'normal'):

                    F += - (1-self.beta)*inner(self.G1+self.G1.T,self.zeta)*dx

                if (self.constitutive_equation == 'Giesekus'):

                    if (self.constitutive_type == 'conf'):

                        F += (self.gmf*self.Wi2/(1-self.beta2))*inner((self.tau0-self.Id)*(self.tau0-self.Id),self.zeta)*dx \

                    elif (self.constitutive_type == 'normal'):

                        F += (self.gmf*self.Wi2/(1-self.beta2))*inner(self.tau0*self.tau0,self.zeta)*dx \
            
            elif (self.axisymmetry == 'Axi'):

                F += (1/self.dt)*inner(self.Wi*(self.tau-self.tau0),self.zeta)*self.x_axi[0]*dx\
                    + 0.5*(inner(dot(self.Wi*self.u0,nabla_grad(self.tau)),self.zeta)*self.x_axi[0]*dx \
                    - inner(dot(self.Wi*self.tau, trans(self.G1)) \
                    - dot(self.Wi*self.G1, self.tau),self.zeta)*self.x_axi[0]*dx) \
                    + 0.5*(inner(dot(self.Wi*self.u0,nabla_grad(self.tau0)),self.zeta)*self.x_axi[0]*dx\
                    - inner(dot(self.Wi*self.tau0, trans(self.G1)) \
                    - dot(self.Wi*self.G1, self.tau0),self.zeta)*self.x_axi[0]*dx) \

                Ftt = (1/self.dt)*self.Wi*(self.tautt-self.tau0tt)*self.zetatt*self.x_axi[0]*dx \
                    + 0.5*(self.tautt + self.Wi*self.u0[1]*self.tautt.dx(1) + self.Wi*self.u0[0]*self.tautt.dx(0) \
                    - 2*self.Wi*self.tautt*self.G1_tt )*self.zetatt*self.x_axi[0]*dx \
                    + 0.5*(self.tau0tt + self.Wi*self.u0[1]*self.tau0tt.dx(1) + self.Wi*self.u0[0]*self.tau0tt.dx(0) \
                    - 2*self.Wi*self.tau0tt*self.G1_tt)*self.zetatt*self.x_axi[0]*dx \

                if (self.ps_element == 'DG'):

                    F += self.Wi*0.5*inner((flux(self.u0,self.fn,self.tau)('+') - flux(self.u0,self.fn,self.tau)('-')),jump(self.zeta))*self.x_axi[0]*dS \
                        + self.Wi*0.5*inner((flux(self.u0,self.fn,self.tau0)('+') - flux(self.u0,self.fn,self.tau0)('-')),jump(self.zeta))*self.x_axi[0]*dS

                if (self.stability == 'SUPG'):

                    h = 1*CellDiameter(self.mesh)
                    supg = (h/magnitude(self.u0)+0.000001)

                    F += 0.5*self.Wi*inner(dot(self.u0, nabla_grad(self.zeta)), supg*dot(self.u0,nabla_grad(self.tau)))*self.x_axi[0]*dx \
                      + 0.5*self.Wi*inner(dot(self.u0, nabla_grad(self.zeta)), supg*dot(self.u0,nabla_grad(self.tau0)))*self.x_axi[0]*dx \

                    Ftt += 0.5*self.Wi*inner((self.u0[1]*self.zetatt.dx(1) + self.u0[0]*self.zetatt.dx(0)), supg*(self.u0[1]*self.tautt.dx(1) + self.u0[0]*self.tautt.dx(0)))*self.x_axi[0]*dx \
                        + 0.5*self.Wi*inner((self.u0[1]*self.zetatt.dx(1) + self.u0[0]*self.zetatt.dx(0)), supg*(self.u0[1]*self.tau0tt.dx(1) + self.u0[0]*self.tau0tt.dx(0)))*self.x_axi[0]*dx \

                if (self.constitutive_type == 'normal'):

                    F += - (1-self.beta)*inner(self.G1+trans(self.G1),self.zeta)*self.x_axi[0]*dx \

                    Ftt += -2*(1-self.beta)*self.G1_tt*self.zetatt*self.x_axi[0]*dx \
            
                if (self.constitutive_equation == 'Giesekus'):

                    if (self.constitutive_type == 'conf'):

                        F += (self.gmf)*inner((self.tau0-self.Id)*(self.tau0-self.Id),self.zeta)*self.x_axi[0]*dx \

                        Ftt += (self.gmf)*(self.tau0tt-1)*(self.tau0tt-1)*self.zetatt*self.x_axi[0]*dx \

                    elif (self.constitutive_type == 'normal'):

                        eps_ptt = 0.05

                        F += 0.5*inner((exp((eps_ptt*self.Wi*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/(1-self.beta))-1)*self.tau0,self.zeta)*self.x_axi[0]*dx \
                          + 0.5*inner((exp((eps_ptt*self.Wi*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/(1-self.beta))-1)*self.tau,self.zeta)*self.x_axi[0]*dx \
                        
                        # F += 0.5*(xi_ptt*self.lamb*inner((self.GRAD+trans(self.GRAD))*self.tau0 + self.tau0*(self.GRAD+trans(self.GRAD)),self.zeta)*self.x_axi[0]*dx) \
                        #     + 0.5*(xi_ptt*self.lamb*inner((self.GRAD+trans(self.GRAD))*self.tau + self.tau*(self.GRAD+trans(self.GRAD)),self.zeta)*self.x_axi[0]*dx) \
                        
                        #(self.gmf*self.lamb/(self.eta_p))*inner(self.tau0*self.tau0,self.zeta)*self.x_axi[0]*dx \

                        Ftt += 0.5*inner((exp((eps_ptt*self.Wi*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/(1-self.beta))-1)*self.tau0tt,self.zetatt)*self.x_axi[0]*dx \
                            + 0.5*inner((exp((eps_ptt*self.Wi*(self.tau0[0,0]+self.tau0[1,1]+self.tau0tt))/(1-self.beta))-1)*self.tautt,self.zetatt)*self.x_axi[0]*dx \
                        # Ftt += 0.5*(xi_ptt*self.lamb*inner((2*self.u0[0]/self.x_axi[0])*self.tau0tt + self.tau0tt*(2*self.u0[0]/self.x_axi[0]),self.zetatt)*self.x_axi[0]*dx) \
                        #     + 0.5*(xi_ptt*self.lamb*inner((2*self.u0[0]/self.x_axi[0])*self.tautt + self.tautt*(2*self.u0[0]/self.x_axi[0]),self.zetatt)*self.x_axi[0]*dx) \
                        
                        #(self.gmf*self.lamb/(self.eta_p))*self.tau0tt*self.tau0tt*self.zetatt*self.x_axi[0]*dx \


        self.a_ce = lhs(F)
        self.m_ce = rhs(F)

        self.A_ce = PETScMatrix()
        self.M_ce = PETScVector()

        if (self.axisymmetry == 'Axi'):

            self.a_tt = lhs(Ftt)
            self.m_tt = rhs(Ftt)

            self.A_tt = PETScMatrix()
            self.M_tt = PETScVector()

    """Construct weak form for navier stokes equations (IPCS scheme)"""
    def ns_form(self):

        if (self.method == 'Cons'):

            if (self.axisymmetry == 'Axi'):

                outn = as_vector([ngamma(self.phi0)[0],0,ngamma(self.phi0)[1]])

                out = outer(outn, outn)

                curv_term = self.curvature*mgrad(self.phi0)*inner((Identity(3) \
                        - out), epsilon_axi(self.v,self.x_axi))*self.x_axi[0]*dx

            elif (self.axisymmetry == 'No'):

                curv_term = self.curvature*mgrad(self.phi0)*inner((self.Id \
                        - outer(ngamma(self.phi0), ngamma(self.phi0))), epsilon(self.v))*dx

        if (self.fluid == 'Newtonian' or self.fluid == 'GNF'):

            if (self.dimensional == 'Dim'):

                if (self.axisymmetry == 'Axi'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho*self.u0, self.v)*self.x_axi[0]*dx \
                        + inner(dot(self.rho*self.u0, nabla_grad(self.u)), self.v)*self.x_axi[0]*dx \
                        + Constant(2.0)*inner(self.mu*epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                        - self.p0*div_axi(self.v,self.x_axi)*self.x_axi[0]*dx \
                        + inner(self.rho*self.g,self.v)*self.x_axi[0]*dx \

                elif (self.axisymmetry == 'No'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho*self.u0, self.v)*dx \
                        + inner(dot(self.rho*self.u0, nabla_grad(self.u)), self.v)*dx \
                        + Constant(2.0)*inner(self.mu*epsilon(self.u), epsilon(self.v))*dx \
                        - self.p0*div(self.v)*dx \
                        + inner(self.rho*self.g,self.v)*dx \

            elif (self.dimensional == 'NonDim'):

                if (self.axisymmetry == 'Axi'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho*self.u0, self.v)*self.x_axi[0]*dx \
                        + inner(dot(self.rho*self.u0, nabla_grad(self.u)), self.v)*self.x_axi[0]*dx \
                        + (Constant(2.0))*inner(self.mu*epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                        - self.p0*div_axi(self.v,self.x_axi)*self.x_axi[0]*dx \
                        + (1/(self.Fr**2))*inner(self.rho*self.g,self.v)*self.x_axi[0]*dx \

                elif (self.axisymmetry == 'No'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho*self.u0, self.v)*self.x_axi[0]*dx \
                        + inner(dot(self.rho*self.u0, nabla_grad(self.u)), self.v)*self.x_axi[0]*dx \
                        + (Constant(2.0)*self.mu/self.Re)*inner(epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                        - self.p0*div_axi(self.v,self.x_axi)*self.x_axi[0]*dx \
                        + (1/(self.Fr**2))*inner(self.rho*self.g,self.v)*self.x_axi[0]*dx \

        elif (self.fluid == 'Viscoelastic'):

            if (self.dimensional == 'Dim'):

                if (self.axisymmetry == 'Axi'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho*self.u0, self.v)*self.x_axi[0]*dx \
                        + inner(dot(self.rho*self.u0, nabla_grad(self.u)), self.v)*self.x_axi[0]*dx \
                        - self.p0*div_axi(self.v,self.x_axi)*self.x_axi[0]*dx \
                        + inner(self.rho*self.g,self.v)*self.x_axi[0]*dx \

                    if (self.DEVSSG == 'True'):

                        self.G3D = as_tensor([[self.G1[0,0],0, self.G1[0,1]],
                                              [0,self.G1_tt,0],
                                              [self.G1[1,0],0,self.G1[1,1]]])

                        ns1 += Constant(2.0)*inner(self.eta_s*epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                            + 1*Constant(2.0)*inner((1-self.eta_s)*epsilon_axi(self.u0, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                            - 1*inner((1-self.eta_s)*(self.G3D+trans(self.G3D)),epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                    
                    elif (self.DEVSSG != 'True'):

                        ns1 += Constant(2.0)*inner((self.eta_s+self.eta_p)*epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                            - Constant(2.0)*inner((self.eta_p)*epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                   
                if (self.constitutive_equation == 'OB' or self.constitutive_equation == 'Giesekus'):

                        if (self.constitutive_type == 'normal'):

                            self.tau_axi = as_tensor([[self.tau0[0,0], 0, self.tau0[1,0]],
                                                        [0, self.tau0tt, 0],
                                                        [self.tau0[1,0], 0, self.tau0[1,1]]])

                            ns1 += self.c*inner(self.tau_axi, nabla_grad_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                                
                        elif (self.constitutive_type == 'conf'):

                            self.tau_axi = as_tensor([[self.tau0[0,0] - 1, 0, self.tau0[1,0]],
                                                        [0, self.tau0tt - 1, 0],
                                                        [self.tau0[1,0], 0, self.tau0[1,1] - 1]])


                            ns1 += (self.c*(self.eta_p)/(self.lamb))*inner(self.tau_axi, nabla_grad_axi(self.v, self.x_axi))*self.x_axi[0]*dx \

                elif (self.axisymmetry == 'No'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho0*self.u0, self.v)*dx \
                        + inner(dot(self.rho*self.u0, grad(self.u)), self.v)*dx \
                        + Constant(2.0)*inner(self.eta_s*epsilon(self.u), epsilon(self.v))*dx \
                        - self.p0*div(self.v)*dx \
                        + inner(self.rho*self.g,self.v)*dx \

                    if (self.constitutive_equation == 'OB' or self.constitutive_equation == 'Giesekus'):

                        if (self.constitutive_type == 'normal'):

                            ns1 += self.c*inner(self.tau0, grad(self.v))*dx \

                        elif (self.constitutive_type == 'conf'):

                            ns1 += (self.eta_p/self.lamb)*inner(self.tau0-self.Id, grad(self.v))*dx \

            if (self.dimensional == 'NonDim'):

                if (self.axisymmetry == 'Axi'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho*self.u0, self.v)*self.x_axi[0]*dx \
                        + inner(dot(self.rho*self.u0, nabla_grad(self.u)), self.v)*self.x_axi[0]*dx \
                        - self.p0*div_axi(self.v,self.x_axi)*self.x_axi[0]*dx \
                        + (1/(self.Fr))*inner(self.rho*self.g,self.v)*self.x_axi[0]*dx \

                    if (self.DEVSSG == 'True'):

                        self.G3D = as_tensor([[self.G1[0,0],0, self.G1[1,0]],
                                            [0,self.G1_tt,0],
                                            [self.G1[0,1],0,self.G1[1,1]]])

                        ns1 += (Constant(2.0)*self.beta/self.Re)*inner(epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                            + (Constant(2.0)*(self.theta)/self.Re)*inner(epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                            - inner(((self.theta)/self.Re)*(self.G3D+trans(self.G3D)),epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \
                    
                    else:

                        ns1 += inner((Constant(2.0)*self.beta/self.Re)*epsilon_axi(self.u, self.x_axi), epsilon_axi(self.v, self.x_axi))*self.x_axi[0]*dx \


                    if (self.constitutive_equation == 'OB' or self.constitutive_equation == 'Giesekus'):

                        if (self.constitutive_type == 'normal'):

                            self.tau_axi = as_tensor([[self.tau0[0,0], 0, self.tau0[1,0]],
                                                        [0, self.tau0tt, 0],
                                                        [self.tau0[1,0], 0, self.tau0[1,1]]])

                            ns1 += (self.c/self.Re2)*inner(self.tau_axi, nabla_grad_axi(self.v, self.x_axi))*self.x_axi[0]*dx \

                        elif (self.constitutive_type == 'conf'):

                            self.tau_axi = as_tensor([[self.tau0[0,0] - 1, 0, self.tau0[1,0]],
                                                        [0, self.tau0tt - 1, 0],
                                                        [self.tau0[1,0], 0, self.tau0[1,1] - 1]])

                            ns1 += inner(((1-self.beta)/(self.Re2*self.Wi))*self.tau_axi, nabla_grad_axi(self.v, self.x_axi))*self.x_axi[0]*dx \

                elif (self.axisymmetry == 'No'):

                    ns1 = (1/self.dt)*inner(self.rho*self.u - self.rho0*self.u0, self.v)*dx \
                        + inner(dot(self.rho*self.u0, grad(self.u)), self.v)*dx \
                        + (Constant(2.0)*self.beta/self.Re)*inner(epsilon(self.u), epsilon(self.v))*dx \
                        - self.p0*div(self.v)*dx \
                        + (1/(self.Fr))*inner(self.rho*self.g,self.v)*dx \

                    if (self.constitutive_equation == 'OB' or self.constitutive_equation == 'Giesekus'):

                        if (self.constitutive_type == 'normal'):

                            ns1 += (self.c/self.Re2)*inner(self.tau0, grad(self.v))*dx \

                        elif (self.constitutive_type == 'conf'):

                            ns1 += ((1-self.beta)/(self.Re2*self.Wi))*inner(self.tau0-self.Id, grad(self.v))*dx \

        if (self.axisymmetry == 'Axi'):

            ns2 = (1/self.rho)*(dot(nabla_grad(self.p),nabla_grad(self.q)) \
                - dot(nabla_grad(self.p0),nabla_grad(self.q)))*self.x_axi[0]*dx \
                + (1/self.dt)*div_axi(self.u_, self.x_axi)*self.q*self.x_axi[0]*dx \

            ns3 = inner(self.u,self.v)*self.x_axi[0]*dx - inner(self.u_,self.v)*self.x_axi[0]*dx \
                + (self.dt/self.rho)*inner(nabla_grad(self.p_-self.p0),self.v)*self.x_axi[0]*dx

        elif (self.axisymmetry == 'No'):

            ns2 = (1/self.rho)*(dot(nabla_grad(self.p),nabla_grad(self.q)) \
                - dot(nabla_grad(self.p0),nabla_grad(self.q)))*dx \
                + (1/self.dt)*div(self.u_)*self.q*dx \

            ns3 = inner(self.u,self.v)*dx - inner(self.u_,self.v)*dx \
                + (self.dt/self.rho)*inner(nabla_grad(self.p_-self.p0),self.v)*dx

        ns1 += curv_term

        self.a_ns1 = lhs(ns1)
        self.m_ns1 = rhs(ns1)

        self.A_ns1 = PETScMatrix()
        self.M_ns1 = PETScVector()

        self.a_ns2 = lhs(ns2)
        self.m_ns2 = rhs(ns2)

        self.A_ns2 = PETScMatrix()
        self.M_ns2 = PETScVector()

        self.a_ns3 = lhs(ns3)
        self.m_ns3 = rhs(ns3)

        self.A_ns3 = PETScMatrix()
        self.M_ns3 = PETScVector()


    """Solve the level set equation"""
    def ls_solve(self, phi):

        assemble(self.a_ls, tensor = self.A_ls)
        assemble(self.m_ls, tensor = self.M_ls)

        solve(self.A_ls, phi.vector(), self.M_ls, self.ls_linear_solver, self.ls_preconditioner)

    """Solve the non-conservative level set reinitialisation equation"""
    def nclsr_solve(self, phi0):

        # phi0.assign(phi)

        self.sign = sgn(phi0, self.eps1)

        self.E = 0
        self.tol = 1.0e-4
        
        for self.n in range(self.rein_steps):
            
            solve(self.F_nclsr == 0, self.phi_rein, solver_parameters={"newton_solver": {"linear_solver": 'gmres', \
                                               "preconditioner": 'default', "maximum_iterations": 20, \
                                               "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
                                               form_compiler_parameters={"optimize": True})


            # assemble(self.a_ncf, tensor = self.A_ncf)
            # assemble(self.m_ncf, tensor = self.M_ncf)

            # solve(self.A_ncf, phi_rein.vector(), self.M_ncf, 'gmres', 'default')

            # error = (((phi_rein - phi0)/self.dtau)**2)*dx
            # self.E = sqrt(assemble(error))

            # if self.E < self.tol:
            #     break

            phi0.assign(self.phi_rein)

        # phi.assign(phi_rein)

    """Solve the conservative level set reinitialisation equation"""
    def clsr_solve(self, phi0):

        # assemble(self.a_nf, tensor = self.A_nf)
        # assemble(self.m_nf, tensor = self.M_nf)

        # solve(self.A_nf, self.phin.vector(), self.M_nf, 'gmres', 'default')

        # solve(self.a_nf == self.m_nf, self.phin)

        
        solve(self.F_nf == 0, self.phin, solver_parameters={"newton_solver": {"linear_solver": 'gmres', \
                                    "preconditioner": 'default', "maximum_iterations": 20, \
                                    "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
                                    form_compiler_parameters={"optimize": True})

        # self.phin = ngamma(self.phi0)

        self.E = 0
        self.tol = 1.0e-4
        
        for self.n in range(self.rein_steps):
            
            solve(self.F_clsr == 0, self.phi_rein, solver_parameters={"newton_solver": {"linear_solver": 'gmres', \
                                            "preconditioner": 'default', "maximum_iterations": 20, \
                                            "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
                                            form_compiler_parameters={"optimize": True})

            # assemble(self.a_cf, tensor = self.A_cf)
            # assemble(self.m_cf, tensor = self.M_cf)

            # solve(self.A_cf, self.phi_rein.vector(), self.M_cf, 'gmres', 'default')

            # error = (((phi_rein - phi0)/self.dtau)**2)*dx
            # self.E = sqrt(assemble(error))

            # if self.E < self.tol:
            #     break

            phi0.assign(self.phi_rein)

        # phi.assign(phi_rein)

    """"Solve DEVSSG equation"""
    def DEVSSG_solve(self):

        assemble(self.a_DEVSSG, tensor = self.A_DEVSSG)
        assemble(self.m_DEVSSG, tensor = self.M_DEVSSG)

        solve(self.A_DEVSSG, self.G1.vector(), self.M_DEVSSG, self.constitutive_equation_linear_solver, self.constitutive_equation_preconditioner)

        assemble(self.a_DEVSSG_tt, tensor = self.A_DEVSSG_tt)
        assemble(self.m_DEVSSG_tt, tensor = self.M_DEVSSG_tt)

        solve(self.A_DEVSSG_tt, self.G1_tt.vector(), self.M_DEVSSG_tt, self.constitutive_equation_linear_solver, self.constitutive_equation_preconditioner)

        self.G3D = as_tensor([[self.G1[0,0],0, self.G1[1,0]],
                    [0,self.G1_tt,0],
                    [self.G1[0,1],0,self.G1[1,1]]])

    """Solve the Oldroyd-b equation."""
    def constitutive_equation_solve(self):

        assemble(self.a_ce, tensor = self.A_ce)
        assemble(self.m_ce, tensor = self.M_ce)

        solve(self.A_ce, self.tau.vector(), self.M_ce, self.constitutive_equation_linear_solver, self.constitutive_equation_preconditioner)
        
        if (self.axisymmetry == 'Axi'):

            assemble(self.a_tt, tensor = self.A_tt)
            assemble(self.m_tt, tensor = self.M_tt)

            solve(self.A_tt, self.tautt.vector(), self.M_tt, self.constitutive_equation_linear_solver, self.constitutive_equation_preconditioner)

    """Solve the navier stokes equations"""
    def ns_solve(self, u_, p_):

        outn = as_vector([ngamma(self.phi0)[0],0,ngamma(self.phi0)[1]])

        out = outer(outn, outn)


        self.G3D = as_tensor([[self.G1[0,0],0, self.G1[0,1]],
                        [0,self.G1_tt,0],
                        [self.G1[1,0],0,self.G1[1,1]]])
        
        self.tau_axi = as_tensor([[self.tau0[0,0], 0, self.tau0[1,0]],
                            [0, self.tau0tt, 0],
                            [self.tau0[1,0], 0, self.tau0[1,1]]])

        assemble(self.a_ns1, tensor = self.A_ns1)
        assemble(self.m_ns1, tensor = self.M_ns1)

        for bc in self.bc_ns:
            bc.apply(self.A_ns1)
            bc.apply(self.M_ns1)
            
        solve(self.A_ns1, u_.vector(), self.M_ns1, self.ns_linear_solver, self.ns_preconditioner)

        assemble(self.a_ns2, tensor = self.A_ns2)
        assemble(self.m_ns2, tensor = self.M_ns2)

        solve(self.A_ns2, p_.vector(), self.M_ns2, self.ns_linear_solver, self.ns_preconditioner)

        assemble(self.a_ns3, tensor = self.A_ns3)
        assemble(self.m_ns3, tensor = self.M_ns3)

        solve(self.A_ns3, u_.vector(), self.M_ns3, self.ns_linear_solver, self.ns_preconditioner)

    """Single-phase form"""
    def sns_form(self):

        devssg_system = inner(self.G-grad(self.us),self.Gs)*dx

        stokes_system = Constant(2.0)*(self.beta+self.theta)*inner(epsilon(self.u),epsilon(self.v))*dx \
                      - self.theta*inner(self.G,grad(self.v))*dx \
                      - self.p*div(self.v)*dx + self.q*div(self.u)*dx \
                      + inner(self.tau1,grad(self.v))*dx \
            
        stress_system = (self.We/self.dt)*inner(self.tau-self.taus, self.S)*dx \
                      + 0.5*(inner(self.tau, self.S)*dx \
                      + self.We*inner(dot(self.us, nabla_grad(self.tau))
                      - dot(self.tau, trans(self.G))
                      - dot(self.G, self.tau), self.S)*dx \
                      - (1-self.beta)*inner(self.G+trans(self.G), self.S)*dx \
                      + self.We*(inner(jump(flux1(self.us,self.nn)*self.tau),  jump(self.S)))*dS) \
                      + 0.5*(inner(self.taus, self.S)*dx \
                      + self.We*inner(dot(self.us, nabla_grad(self.taus))
                      - dot(self.taus, trans(self.G))
                      - dot(self.G, self.taus), self.S)*dx \
                      - (1-self.beta)*inner(self.G+trans(self.G), self.S)*dx \
                      + self.We*(inner(jump(flux1(self.us,self.nn)*self.taus),  jump(self.S)))*dS) \

        self.a_ce = lhs(stress_system)
        self.m_ce = rhs(stress_system)

        self.A_ce = PETScMatrix()
        self.M_ce = PETScVector()

        self.a_st = lhs(stokes_system)
        self.m_st = rhs(stokes_system)

        self.A_st = PETScMatrix()
        self.M_st = PETScVector()

        self.vgp = devssg_system

    def sns_solve(self, w1, tau1):

        solve(self.vgp == 0, self.G, solver_parameters={"newton_solver": {"linear_solver": 'gmres', "preconditioner": 'default',\
                        "maximum_iterations": 20, "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
                            form_compiler_parameters={"optimize": True})

        assemble(self.a_ce, tensor = self.A_ce)
        assemble(self.m_ce, tensor = self.M_ce)

        [bc.apply(self.A_ce,self.M_ce) for bc in self.bcst]

        solve(self.A_ce, tau1.vector(), self.M_ce, 'gmres', 'default')

        assemble(self.a_st, tensor = self.A_st)
        assemble(self.m_st, tensor = self.M_st)

        [bc.apply(self.A_st, self.M_st) for bc in self.bcup]

        solve(self.A_st, w1.vector(), self.M_st)

        self.u1, self.p1 = split(w1)

        # if (self.method == 'NCons'):

        #     curv_term  = self.curvature*mgrad(self.phi0)*inner((self.Id \
        #                - outer(ngamma(self.phi0), ngamma(self.phi0))), epsilon(self.v)) \
        #                * delta_func(self.phi0,self.eps)*dx

    # if (self.constitutive_equation == 'OB-Conf'):

    #     F = (1/self.dt)*inner(tau-tau0,zeta)*dx \
    #       + inner(dot(u0,nabla_grad(tau)) \
    #       - dot(tau, trans(grad(u0))) \
    #       - dot(grad(u0), tau),zeta)*dx \

    #     if (self.dimensional == 'Dim'):

    #         F += (1/self.lamb)*inner(tau - self.Id, zeta)*dx

        # elif (self.dimensional == 'NonDim'):

        #     F += (1/self.Wi)*inner(tau - self.Id, zeta)*dx

    # if (self.constitutive_equation == 'Gie'):

    #     if (self.dimensional == 'Dim'):

    #         pass

    #     elif (self.dimensional == 'NonDim'):

    #         pass

    # if (self.constitutive_equation == 'Gie-Conf'):

    #     if (self.dimensional == 'Dim'):

    #         pass

    #     elif (self.dimensional == 'NonDim'):

    #         pass

    # elif (self.constitutive_equation == 'Giesekus'):

    #     if (self.stability == 'DEVSSG-DG'):

    #         pass

    #     elif (self.stability == None):

    #         F = (1/self.dt)*inner(tau-tau0,zeta)*dx \
    #         + (inner(dot(u0,nabla_grad(tau0)) \
    #         - dot(tau0, trans(grad(u0))) \
    #         - dot(grad(u0), tau0),zeta)*dx \
    #         + (1/Wi)*inner((tau0-self.Id) \
    #         + gmf*(tau0-self.Id)*(tau0-self.Id), zeta)*dx) \
    #         + inner(poly_flux(u0,self.facet_normal,tau0)('+') - poly_flux(u0,self.facet_normal,tau0)('-'),jump(zeta))*dS

            # if (self.axisymmetry == 'Axi'):

            #     F += (1/self.dt)*inner(self.Wi*(self.tau-self.tau0),self.zeta)*self.x_axi[0]*dx\
            #         + 0.5*(inner(dot(self.Wi*self.u0,nabla_grad(self.tau)),self.zeta)*self.x_axi[0]*dx \
            #         - inner(dot(self.Wi*self.tau, trans(nabla_grad(self.u0))) \
            #         - dot(self.Wi*nabla_grad(self.u0), self.tau),self.zeta)*self.x_axi[0]*dx) \
            #         + 0.5*(inner(dot(self.Wi*self.u0,nabla_grad(self.tau0)),self.zeta)*self.x_axi[0]*dx\
            #         - inner(dot(self.Wi*self.tau0, trans(nabla_grad(self.u0))) \
            #         - dot(self.Wi*nabla_grad(self.u0), self.tau0),self.zeta)*self.x_axi[0]*dx) \

            #     Ftt = (1/self.dt)*self.Wi*(self.tautt-self.tau0tt)*self.zetatt*self.x_axi[0]*dx \
            #         + 0.5*(self.tautt + self.Wi*self.u0[1]*self.tautt.dx(1) + self.Wi*self.u0[0]*self.tautt.dx(0) \
            #         - 2*self.Wi*(self.tautt*self.u0[0])/self.x_axi[0] )*self.zetatt*self.x_axi[0]*dx \
            #         + 0.5*(self.tau0tt + self.Wi*self.u0[1]*self.tau0tt.dx(1) + self.Wi*self.u0[0]*self.tau0tt.dx(0) \
            #         - 2*self.Wi*self.tau0tt*self.u0[0]/self.x_axi[0])*self.zetatt*self.x_axi[0]*dx \

            #     if (self.constitutive_type == 'normal'):

            #         F += - (1-self.beta)*inner(nabla_grad(self.u0)+nabla_grad(self.u0).T,self.zeta)*self.x_axi[0]*dx \
            
            #     if (self.constitutive_equation == 'Giesekus'):

            #         if (self.constitutive_type == 'conf'):

            #             F += (self.gmf*self.Wi/(1-self.beta))*inner((self.tau0-self.Id)*(self.tau0-self.Id),self.zeta)*self.x_axi[0]*dx \

            #         elif (self.constitutive_type == 'normal'):

            #             F += (self.gmf*self.Wi/(1-self.beta))*inner(self.tau0*self.tau0,self.zeta)*self.x_axi[0]*dx \

            # elif (self.axisymmetry == 'No'):

                        # if (self.axisymmetry == 'Axi'):

            #     F += (1/self.dt)*inner(self.lamb*(self.tau-self.tau0),self.zeta)*self.x_axi[0]*dx\
            #         + 0.5*(inner(dot(self.lamb*self.u0,nabla_grad(self.tau))\
            #         - dot(self.lamb*self.tau, trans(nabla_grad(self.u0))) \
            #         - dot(self.lamb*nabla_grad(self.u0), self.tau),self.zeta)*self.x_axi[0]*dx) \
            #         + 0.5*(inner(dot(self.lamb*self.u0,nabla_grad(self.tau0))\
            #         - dot(self.lamb*self.tau0, trans(nabla_grad(self.u0))) \
            #         - dot(self.lamb*nabla_grad(self.u0), self.tau0),self.zeta)*self.x_axi[0]*dx) \
            #         # + 0.5*inner(flux(self.u0,self.fn,self.tau0)('+') - flux(self.u0,self.fn,self.tau0)('-'),jump(self.zeta))*self.x_axi[0]*dS \
            #         # + 0.5*inner(flux(self.u0,self.fn,self.tau)('+') - flux(self.u0,self.fn,self.tau)('-'),jump(self.zeta))*self.x_axi[0]*dS

            #     if (self.constitutive_type == 'normal'):

            #         F += - self.eta_p*inner(nabla_grad(self.u0)+nabla_grad(self.u0).T,self.zeta)*self.x_axi[0]*dx \
            
            #     if (self.constitutive_equation == 'Giesekus'):

            #         F += (self.gmf*self.lamb/self.eta_p_out)*inner(self.tau0*self.tau0,self.zeta)*self.x_axi[0]*dx \
                
            #     Ftt = (1/self.dt)*self.lamb*(self.tautt-self.tau0tt)*self.zetatt*self.x_axi[0]*dx \
            #         + 0.5*(self.tautt + self.lamb*self.u0[1]*self.tautt.dx(1) + self.lamb*self.u0[0]*self.tautt.dx(0) \
            #         - 2*self.lamb*(self.tautt*self.u0[0])/self.x_axi[0] )*self.zetatt*self.x_axi[0]*dx \
            #         + 0.5*(self.tau0tt + self.lamb*self.u0[1]*self.tau0tt.dx(1) + self.lamb*self.u0[0]*self.tau0tt.dx(0) \
            #         - 2*self.lamb*self.tau0tt*self.u0[0]/self.x_axi[0])*self.zetatt*self.x_axi[0]*dx \
            #         # + 0.5*inner(flux(self.u0,self.fn,self.tau0tt)('+') - flux(self.u0,self.fn,self.tau0tt)('-'),jump(self.zetatt))*self.x_axi[0]*dS \
            #         # + 0.5*inner(flux(self.u0,self.fn,self.tautt)('+') - flux(self.u0,self.fn,self.tautt)('-'),jump(self.zetatt))*self.x_axi[0]*dS

            #     if (self.constitutive_type == 'normal'):

            #         Ftt += - (2*self.eta_p*self.u0[0]/self.x_axi[0])*self.zetatt*self.x_axi[0]*dx \

            #     if (self.constitutive_equation == 'Giesekus'):

            #         Ftt += (self.gmf*self.lamb/self.eta_p_out)*self.tau0tt*self.tau0tt*self.zetatt*self.x_axi[0]*dx \

            # elif (self.axisymmetry == 'No'):