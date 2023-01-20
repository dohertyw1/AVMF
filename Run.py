from tqdm import tqdm
from Multiphase import *
from Funcs import *
from Process import *
from ParameterHandler import *

class Run(Multiphase, Flow, Process, ParameterHandler):

    def __init__(self, phase, parameters_file, strings_file): #, parameters_file, strings_file, eta_s_outer, eta_p_outer, extra):
        
        self.phase=phase
        set_log_active(False)
        self.comm = MPI.comm_world
        self.rank = MPI.rank(self.comm)
        self.parameters_file = parameters_file
        self.strings_file = strings_file

        parameters['ghost_mode'] = 'shared_facet' 

        if not (self.phase  == 'single'):

            self.read_parameters(self.parameters_file, self.strings_file)

            # self.eta_s_out = eta_s_outer
            # self.eta_p_out = eta_p_outer
            # self.rein_steps = ls_rein_steps
            
            # self.sizex = sizex
            # self.sizey = sizey
            # self.extra = extra
            # self.radius = radius

            # self.gmf = gmf

            # self.Wi2 = wi
            # self.Wi2 = Wi
            # self.Re2 = Re_outer
            # self.curvature = 1/self.Eo
            # self.extra = extra

            Flow.__init__(self)
            Process.__init__(self)

            self.file_string = f'{self.method}_{self.fluid}_{self.element}_{self.dimensional}_{self.extra}'

    def single_phase_run(self):

        self.We = 0.8
        self.beta = 0.59
        self.theta = 0.41
        self.T = 8
        self.num_steps = 8000
        self.dt = self.T/self.num_steps
        self.single_phase_mesh = 'godzilla.xml'
        self.drag_list = []

        self.phase = 'single'
        self.mesh = Mesh(f'{self.single_phase_mesh}')
        self.nn = FacetNormal(self.mesh)
        self.ns_funcs()
        self.bcs()
        self.sns_form()

        self.t = 0
        self.n = 0

        for self.n in tqdm(range(int(self.num_steps))):

            # self.us, self.ps = split(self.ws)

            self.sns_solve(self.w1, self.tau1)

            self.sns_drag(self.p1, self. u1, self. tau1)

            self.sns_drag_to_file('sns_test')

            self.taus.assign(self.tau1)
            self.ws.assign(self.w1)


    def newtonian_run(self):

        self.set_up_files()
        
        self.mesh_constructor()
        self.level_set()
        self.level_set_funcs()
        self.ns_funcs()
        self.bcs()

        self.ls_form()
        self.n_form()
        # self.phin = Function(self.Q_normal)

        if (self.method == 'NCons'):

            self.nclsr_form()

        elif (self.method == 'Cons'):

            self.clsr_form()

        self.phi = Function(self.Q)
        # self.phi_rein = Function(self.Q_rein)
        
        self.mat_pams()

        self.ns_form()

        self.t = 0
        self.n = 0
        self.q = 0
        self.k = 0

        # File('phio.pvd') << self.phi0

        # for self.k in range(int(8)):

        #     if (self.method == 'NCons'):

        #         self.nclsr_solve(self.phi0)

        #     elif (self.method == 'Cons'):

        #         # self.clsr_form(self.compressor, self.eps)

        #         self.clsr_solve(self.phi0)

        #         if self.rank ==0:
        #             print('done')

        for self.n in tqdm(range(int(self.num_steps))):

            # self.compressor = 1
            # self.eps = Constant((1.5/4)*self.hmin)
            # # self.rein_steps = 1
            # self.dtau = Constant(0.5*self.hmin**1.075) 

            self.t += self.dt

            if (self.element == 'CG'):

                self.ls_solve(self.phi)
                self.phi0.assign(self.phi)

            elif (self.element == 'DG'):

                self.ls_solve(self.phi_h)
                self.phi0.assign(self.phi_h)

            if (self.n % self.rein_div == 0):

                if (self.method == 'NCons'):

                    self.nclsr_solve(self.phi0)

                elif (self.method == 'Cons'):

                    # self.clsr_form(self.compressor, self.eps)

                    self.clsr_solve(self.phi0)

            self.ns_solve(self.u_, self.p_)
            self.u0.assign(self.u_)
            self.p0.assign(self.p_)

            # if (self.q % 100 == 0):

            #     self.process_shape(self.t)

            if (self.write_bool == 'True'):

                if (self.q % self.q_div == 0):

                    self.write_to_file()

                if (self.q == 0):

                    self.write_parameters()

                self.process_data()

            self.q += 1

        if (self.rank == 0):

            print(f"{bcolors.OKGREEN}Finished the simulation!\nResults saved in Two_Phase/{self.file_string}.{bcolors.ENDC}")
            
        if (self.write_bool == 'True'):

            self.process_shape(self.t)

            self.post_process()

    def viscoelastic_run(self):

        self.set_up_files()

        self.mesh_constructor()
        self.level_set()
        self.level_set_funcs()
        self.ns_funcs()
        self.bcs()

        self.ls_form()
        self.n_form()

        if (self.method == 'NCons'):

            self.nclsr_form()

        elif (self.method == 'Cons'):

            self.clsr_form()

        self.phi = Function(self.Q)
        
        self.mat_pams()
        self.DEVSSG_form()



        self.ns_form()
        self.constitutive_equation_form()

        self.tau = Function(self.T)
        self.tautt = Function(self.Ttheta)
        self.G = Function(self.T_DEVSSG)

        self.t = 0
        self.n = 0
        self.q = 0
        self.k = 0

        for self.k in range(int(8)):

            self.clsr_solve(self.phi0)

        for self.n in tqdm(range(int(self.num_steps))):

            # self.compressor = 1
            # self.eps = Constant((1.5/4)*self.hmin)

            self.t += self.dt

            if (self.element == 'CG'):

                self.ls_solve(self.phi)
                self.phi0.assign(self.phi)

            elif (self.element == 'DG'):

                self.ls_solve(self.phi_h)
                self.phi0.assign(self.phi_h)

            if (self.n % self.rein_div == 0):

                if (self.method == 'NCons'):

                    self.nclsr_solve(self.phi0)

                elif (self.method == 'Cons'):

                    self.clsr_solve(self.phi0)

            # File('phi0.pvd') << self.phi0

            self.ns_solve(self.u_, self.p_)

            self.u0.assign(self.u_)
            self.p0.assign(self.p_)

            if (self.DEVSSG == 'True'):

                self.DEVSSG_solve()
            #     self.GRAD.assign(self.G1)
            #     self.GRAD_tt.assign(self.G1_tt)
            
            # elif (self.DEVSSG != 'True' and self.axisymmetry == 'Axi'):

            #     self.GRAD = grad(self.u0)
            #     self.GRAD_tt = self.u0[0]/self.x_axi[0]

            self.constitutive_equation_solve()

            self.tau0.assign(self.tau)
            self.tau0tt.assign(self.tautt)
                
            # self.u0.assign(self.u_)
            # self.p0.assign(self.p_)

            # if (self.q % 2 == 0):

            #     self.process_shape(self.t)

            if (self.write_bool == 'True'):

                if (self.q % self.q_div == 0):

                    self.write_to_file()

                if (self.q == 0):

                    self.write_parameters()

                self.process_data()

            self.q += 1

        if (self.rank == 0):

            print(f"{bcolors.OKGREEN}Finished the simulation!\nResults saved in Two_Phase/{self.file_string}.{bcolors.ENDC}")
        
        if (self.write_bool == 'True'):

            self.process_shape(self.t)

            self.post_process()

    def execute(self):

        if (self.phase == 'single'):

            self.single_phase_run()

        else:

            if (self.fluid == 'Newtonian'):

                self.newtonian_run()

            elif (self.fluid == 'Viscoelastic'):

                self.viscoelastic_run()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius=0.1,
#                      extra = '10_06_22_1')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius=0.15,
#                      extra = '10_06_22_15')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius=0.2,
#                      extra = '10_06_22_2')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius=0.25,
#                      extra = '10_06_22_25')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      sizex = 60,
#                      sizey = 100,
#                      extra = '60x100')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      sizex = 20,
#                      sizey = 100,
#                      extra = 'dimtestlol',
#                      radius = 0.00212)

# Viscoelastic_D.execute()

Viscoelastic_D = Run(phase='two',
                    parameters_file = 'Viscoelastic_D.txt',
                     strings_file = 'Viscoelastic_D_strings.txt')

Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      sizex = 120,
#                      sizey = 200,
#                      extra = '120x200')

# Viscoelastic_D.execute()

# Newtonian_D = Run(phase='two',parameters_file = 'Newtonian_D_case2.txt',
#                      strings_file = 'Newtonian_D_strings.txt')


# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.1,
#                      extra = '0.1')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.125,
#                      extra = '0.125')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.175,
#                      extra = '0.175')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.2,
#                      extra = '0.2')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.225,
#                      extra = '0.225')

# Viscoelastic_D.execute()




# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.275,
#                      extra = '0.275')

# Viscoelastic_D.execute()

# Viscoelastic_D = Run(phase='two',
#                     parameters_file = 'Viscoelastic_D.txt',
#                      strings_file = 'Viscoelastic_D_strings.txt',
#                      radius = 0.3,
#                      extra = '0.3')

# Viscoelastic_D.execute()
