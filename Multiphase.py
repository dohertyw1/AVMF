from fenics import *
import meshio
from Flow import *
from Funcs import *
from Process import *
from ParameterHandler import *

class Multiphase():

    """"Initialise material parameters, mesh and test case data"""
    def __init__(self):

        pass
                   
    """Construct the mesh for the problem."""
    def mesh_constructor(self):

        if (self.dimension == '2D'):

            if (self.mesh_from_file == 'False'):

                self.mesh = RectangleMesh(Point(self.cox1,self.coy1),Point(self.cox2,self.coy2),self.sizex,self.sizey, self.element_orientation) 
                self.amesh =  RectangleMesh(Point(self.cox1,self.coy1),Point(self.cox2,self.coy2),200,800, self.element_orientation)
                
                # if (self.rank == 0):
                    
                # cell_markers = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
                # cell_markers.set_all(False)
                # for cell in cells(self.mesh):
                #     for facet in facets(cell): 
                #         for vertex in vertices(facet):
                #             if (0.06 <= vertex.point().array()[0] <= 0.07):
                #                 cell_markers[cell] = True

                # self.mesh = refine(self.mesh, cell_markers)

                # cell_markers = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
                # cell_markers.set_all(False)
                # for cell in cells(self.mesh):
                #     for facet in facets(cell): 
                #         for vertex in vertices(facet):
                #             if (0.06 <= vertex.point().array()[0] <= 0.065):
                #                 cell_markers[cell] = True

                # self.mesh = refine(self.mesh, cell_markers)

                # cell_markers = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
                # cell_markers.set_all(False)
                # for cell in cells(self.mesh):
                #     for facet in facets(cell): 
                #         for vertex in vertices(facet):
                #             if (0.06 <= vertex.point().array()[0] <= 0.065):
                #                 cell_markers[cell] = True

                # self.mesh = refine(self.mesh, cell_markers)
                # File(f"{self.file_string}/Meshes/mesh.pvd") << self.mesh

            elif (self.mesh_from_file == 'True'):

                if (self.rank == 0):

                    msh = meshio.read(f"{self.mesh_file_path}")

                    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)

                    try:

                        meshio.write(f"{self.file_string}/mesh.xdmf", triangle_mesh)

                    except FileNotFoundError:

                        os.mkdir(f'{self.file_string}')
                        meshio.write(f"{self.file_string}/mesh.xdmf", triangle_mesh)
                        
                self.mesh = Mesh()
                with XDMFFile(f"{self.file_string}/mesh.xdmf") as infile:
                    infile.read(self.mesh)

        elif (self.dimension == '3D'):

            self.mesh = BoxMesh(Point(self.cox1,self.coy1,self.coz1), Point(self.cox2,self.coy2,self.coz2), self.sizex, self.sizey, self.sizez)
            
        self.fn = FacetNormal(self.mesh)
        self.x_axi = SpatialCoordinate(self.mesh)
        self.hmin = self.mesh.hmin()

    """"Define signed distance function (ncls and cls) and heaviside function (cls)"""
    def level_set(self):

        if (self.method == 'NCons'):

            self.alph = Constant(0.0625*self.hmin)
            # self.eps1 = Constant(1.5*self.hmin)
            self.eps = Constant(1.5*self.hmin) #Constant(self.hmin) 
            self.dtau = Constant(0.1*self.hmin) 
        
        elif (self.method == 'Cons'):

            self.compressor = 4
            self.dtau = Constant(0.1*self.hmin) 
            self.eps = Constant(1.5*self.hmin)
            
        if (self.dimension == '2D'):

            self.sdf = Expression('sqrt((pow((x[0]-A),2))+(pow((x[1]-B),2)))-r', degree=2, A=self.centrex, B=self.centrey, r=self.radius)

            self.g = Constant((0,self.grav))
            
        elif (self.dimension == '3D'):

            self.sdf = Expression('sqrt((pow((x[0]-A),2))+(pow((x[1]-B),2))+(pow((x[2]-C),2)))-r', degree=2,
                                 A=self.centrex, B=self.centrey, C=self.centrez, r=self.radius)

            self.g = Constant((0,self.grav,0))

        if (self.method == 'Cons'):

            self.hdf = Expression('(1/(1+exp((dist/eps))))',degree=2, eps=self.eps, dist=self.sdf)

    def level_set_funcs(self):

        self.Q = FunctionSpace(self.mesh, self.element, self.ls_order)
        self.Q_normal = VectorFunctionSpace(self.mesh, 'CG', self.ls_order)

        if (self.element == 'CG'):

            self.Q_rein = self.Q

        elif (self.element == 'DG'):

            self.Q_rein = FunctionSpace(self.mesh, 'CG', self.ls_order)

        self.phi = TrialFunction(self.Q)
        self.psi = TestFunction(self.Q)

        self.phi_rein = Function(self.Q_rein)
        self.psi_rein = TestFunction(self.Q_rein)

        self.phin = Function(self.Q_normal)
        self.psin = TestFunction(self.Q_normal)

        if (self.element == 'DG'):
        
            self.phi_h = Function(self.Q)

        if (self.method == 'NCons'):

            self.phi0 = interpolate(self.sdf,self.Q)
            self.phiic = interpolate(self.sdf,self.Q)
            self.sign = sgn(self.phi0, self.eps1)
        
        elif (self.method == 'Cons'):

            self.phi0 = interpolate(self.hdf,self.Q)
            self.phiic = interpolate(self.hdf,self.Q)

    def ns_funcs(self):

        if (self.phase == 'single'):

            # self.P = FunctionSpace(mesh, FiniteElement('CG', mesh.ufl_cell(), 1))
            # self.V = FunctionSpace(mesh, VectorElement('CG', mesh.ufl_cell(), 2))
            self.St = FunctionSpace(self.mesh, TensorElement('DG', self.mesh.ufl_cell(), 0))
            self.D = FunctionSpace(self.mesh, TensorElement('CG', self.mesh.ufl_cell(), 2))

            self.stokes = FunctionSpace(self.mesh,
                          MixedElement([ VectorElement('CG', self.mesh.ufl_cell(), 2),
                          FiniteElement('CG', self.mesh.ufl_cell(), 1)]))

            self.tau = TrialFunction(self.St)
            self.S = TestFunction(self.St)
            self.tau1 = Function(self.St)

            (self.u, self.p) = TrialFunctions(self.stokes)
            (self.v, self.q) = TestFunctions(self.stokes)

            self.w1 = Function(self.stokes)
            (self.u1, self.p1) = split(self.w1)

            self.ws = Function(self.stokes)
            (self.us, self.ps) = split(self.ws)

            self.taus = Function(self.St)

            self.G = Function(self.D)
            self.Gs = TestFunction(self.D)

        else:

            """ Pressure spaces/functions """
            self.P = FunctionSpace(self.mesh, 'CG', 1)
            self.p = TrialFunction(self.P)
            self.q = TestFunction(self.P)
            self.p0 = Function(self.P)
            self.p_  = Function(self.P)

            """ Velocity spaces/functions """

            self.V = VectorFunctionSpace(self.mesh, 'CG', 2)

            self.u = TrialFunction(self.V) 
            self.v = TestFunction(self.V)
            self.u0 = Function(self.V) 
            self.u_ = Function(self.V)

            """ Stress spaces/functions """
                
            if (self.ps_element == 'CG'):
                ps_order = 2
            elif (self.ps_element == 'DG'):
                ps_order = 0

            self.T = TensorFunctionSpace(self.mesh,self.ps_element, ps_order)
            self.Ttheta = FunctionSpace(self.mesh,self.ps_element, ps_order)

            self.tau = TrialFunction(self.T)
            self.tautt = TrialFunction(self.Ttheta)
            self.zeta = TestFunction(self.T)
            self.zetatt = TestFunction(self.Ttheta)
            self.tau0 = Function(self.T)
            self.tau0tt = Function(self.Ttheta)

            """DEVSSG space/functions"""
            self.T_DEVSSG = TensorFunctionSpace(self.mesh, 'DG', 2)

            self.G = TrialFunction(self.T_DEVSSG)
            self.R = TestFunction(self.T_DEVSSG)

            self.T_tt_DEVSSG = FunctionSpace(self.mesh, 'DG', 2)
            self.G_tt = TrialFunction(self.T_tt_DEVSSG)
            self.R_tt = TestFunction(self.T_tt_DEVSSG)

            self.G1 = Function(self.T_DEVSSG)
            self.G1_tt = Function(self.T_tt_DEVSSG)
            self.GRAD = Function(self.T_DEVSSG)
            self.GRAD_tt = Function(self.T_tt_DEVSSG)

    """Construct boundary conditions"""
    def bcs(self):

        if not (self.phase == 'single'):
        
            self.bcs = []
            
            # fswalls = f'near(x[0], {self.cox1})'
            # fswalls_x = 'near(x[0], 0) || near(x[0], 1)'
            # fswalls_z = 'near(x[2], 0) || near(x[2], 1)'

            if (self.dimension == '2D'):

                if (self.boundary_conditions == 'freeslip'):

                    walls   = f'near(x[1], {self.coy1}) || near(x[1], {self.coy2})'
                    fswalls = f'near(x[0], {self.cox1}) || near(x[0], {self.cox2})'

                    bcu_noslip  = DirichletBC(self.V, Constant((0, 0)), walls)
                    bcu_fslip  = DirichletBC(self.V.sub(0), Constant(0), fswalls)

                    self.bc_ns = [bcu_noslip, bcu_fslip]
                
                if (self.boundary_conditions == 'noslip'):

                    walls   = f'near(x[1], {self.coy1}) || near(x[1], {self.coy2}) || near(x[0], {self.cox1}) || near(x[0], {self.cox2})'
                    
                    # class Wedge(SubDomain):
                    #     def inside(self, x, on_boundary):
                    #         return on_boundary and (between(x[0], (-1, 1)) and between(x[1], (0, 1)))

                    bcu_noslip  = DirichletBC(self.V, Constant((0, 0)), walls)

                    self.bc_ns = [bcu_noslip]

            elif (self.dimension == '3D'):

                walls = f'near(x[1], {self.coy1}) || near(x[1], {self.coy2})'
                fswall = f'near(x[0], {self.cox1}) || near(x[0], {self.cox2}) || near(x[2], {self.coz1}) || near(x[2], {self.coz2})'

                bcu_noslip  = DirichletBC(self.V, Constant((0, 0, 0)), walls)
                bcu_fslip_x = DirichletBC(self.V.sub(0), Constant(0), fswall)
                # bcu_fslip_z = DirichletBC(self.V.sub(2), Constant(0), fswalls_z)

                self.bc_ns = [bcu_noslip, bcu_fslip_x]
        
        else:

            class Inflow(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[0], -20)

            class Top_Wall(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[1], 2)

            class Bottom_Wall(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[1], 0)

            class Outflow(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[0], 20)

            class Wedge(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and (between(x[0], (-1, 1)) and between(x[1], (0, 1)))

            subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
            
            inflow = Inflow()
            inflow.mark(subdomains, 1)

            top_wall = Top_Wall()
            top_wall.mark(subdomains, 2)

            bottom_wall = Bottom_Wall()
            bottom_wall.mark(subdomains, 3)

            outflow = Outflow()
            outflow.mark(subdomains, 4)

            wedge = Wedge()
            wedge.mark(subdomains, 5)

            self.ds_circle = Measure("ds", subdomain_data=subdomains, subdomain_id=5)

            vel_inflow = Expression(('1.5*(1-(x[1]*x[1]/4))', '0'), degree=2)

            str_inflow = Expression((('2*We*(1-beta)*(9/16)*x[1]*x[1]',
                                        '-3/4*(1-beta)*x[1]'),
                                        ('-3/4*(1-beta)*x[1]', '0')),
                                        degree=2, We=self.We, beta=self.beta)

            stress_inflow_0 = DirichletBC(self.St, str_inflow, subdomains, 1)
            velocity_inflow_1 = DirichletBC(self.stokes.sub(0), vel_inflow, subdomains, 1)
            noslip_top_1 = DirichletBC(self.stokes.sub(0), vel_inflow, subdomains, 2)
            symmetry_1 = DirichletBC(self.stokes.sub(0).sub(1), Constant(0), subdomains, 3)
            outflow_condition_1 = DirichletBC(self.stokes.sub(0).sub(1), Constant(0), subdomains, 4)
            outflow_pressure_1 = DirichletBC(self.stokes.sub(1), Constant(0), subdomains, 4)
            noslip_wedge_1 = DirichletBC(self.stokes.sub(0), Constant((0, 0)), subdomains, 5)

            self.bcst = [stress_inflow_0]

            self.bcup = [velocity_inflow_1,
                    noslip_top_1,
                    symmetry_1,
                    outflow_condition_1,
                    noslip_wedge_1,
                    outflow_pressure_1]
            
            # self.bcup2 = [noslip_top_1,
            #         symmetry_1,
            #         outflow_condition_1,
            #         noslip_wedge_1,
            #         outflow_pressure_1]

    """Calculate density/viscosity functions depending on level set and type of fluid."""
    def mat_pams(self):

        if (self.method == 'NCons'):

            self.rho = phasify_noncon(self.phi0, self.rho1, self.rho2, self.eps)
            self.rho0 = phasify_noncon(self.phi0, self.rho1, self.rho2, self.eps)

            if (self.fluid == 'Newtonian'):

                self.mu = phasify_noncon(self.phi0, self.mu1, self.mu2, self.eps)

            elif (self.fluid == 'Viscoelastic'):

                if (self.constitutive_equation == 'OB' or self.constitutive_equation == 'Giesekus') and (self.dimensional == 'Dim'):

                    self.eta_s = phasify_noncon(self.phi0, self.eta_s_in, self.eta_s_out, self.eps)

                    self.eta_p = phasify_noncon(self.phi0, self.eta_p_in, self.eta_p_out, self.eps)

                    self.lamb = phasify_noncon(self.phi0, self.lamb_in, self.lamb_out, self.eps)

        elif (self.method == 'Cons'):

            self.rho = phasify_con(self.phi0, self.rho1, self.rho2)
            self.rho0 = phasify_con(self.phi0, self.rho1, self.rho2)

            if (self.fluid == 'Newtonian'):

                if (self.dimensional == 'Dim'):

                    self.mu = phasify_con(self.phi0, self.mu1, self.mu2)

                elif (self.dimensional == 'NonDim'):

                    self.mu = phasify_con(self.phi0, self.mu1, self.mu2)

                    self.Re = phasify_con(self.phi0, self.Re1, self.Re2)
            
            elif (self.fluid == 'Viscoelastic'):

                if (self.dimensional == 'Dim'):

                    self.eta_s = phasify_con(self.phi0, self.eta_s_in, self.eta_s_out)

                    self.eta_p = self.eta_p_out #phasify_con(self.phi0, self.eta_p_in, self.eta_p_out)

                    self.lamb = self.lamb_out #phasify_con(self.phi0, self.lamb_in, self.lamb_out)

                    self.c = phasify_con(self.phi0, 0, 1)

                elif (self.constitutive_equation == 'OB' or self.constitutive_equation == 'Giesekus') and (self.dimensional == 'NonDim'):

                    self.beta = phasify_con(self.phi0, self.beta1, self.beta2)

                    self.Re = phasify_con(self.phi0, self.Re2*self.Re_eps, self.Re2)

                    self.Wi = self.Wi2 #phasify_con(self.phi0, self.Wi1, self.Wi2)

                    self.c = phasify_con(self.phi0, 0, 1)

                    self.theta = phasify_con(self.phi0, self.theta_in, self.theta_out)