import numpy as np
import os

class bcolors():

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ParameterHandler():

    def __init__(self):

        pass

    def print_parameters(self):

        print("Inner fluid density is [{}] and outer fluid density is [{}].".format(self.rho1, self.rho2))
        print("Inner fluid viscosity is [{}] and outer fluid viscosity is [{}].".format(self.mu1, self.mu2))
        print("Acceleration due to gravity is: {}.".format(self.grav))
        print("Surface tension coefficient: {}.".format(self.sigma))

    def one_by_one(self):

            try:

                self.rho1, self.rho2 = [float(x) for x in input("Enter the density of the inner fluid then the outer fluid: ").split()]
                self.mu1, self.mu2 = [float(x) for x in input("Enter the viscosity of the inner fluid then the outer fluid: ").split()]

            except ValueError:
                
                print(f"{bcolors.FAIL}Error: Please enter two values with a space in between them.{bcolors.ENDC}")
                quit()

            self.grav = [float(x) for x in input("Enter the aceleration due to gravity: ").split()]
            self.sigma = [float(x) for x in input("Enter the surface tension coefficient: ").split()]

            self.print_parameters()

    def read_in(self, parameters_file, strings_file):

        self.parameters = {}
        self.strings = {}

        try:

            with open(f'parameters/{parameters_file}') as File:

                for line in File:
                    
                    index = line.find(' = ')
                    self.parameters[line[0:index]] = float(line[index+2:].replace('\n', ''))

            with open(f'parameters/{strings_file}') as File:

                for line in File:
                    
                    index = line.find(' = ')
                    self.strings[line[0:index]] = line[index+3:].replace('\n', '')
        
        except FileNotFoundError:

            if (self.rank == 0):

                print(f"{bcolors.FAIL}Error: Parameter file not found. Please ensure it is located in the correct directory.{bcolors.ENDC}")

            quit()

        self.fluid = self.strings['fluid']
        self.dimensional = self.strings['dimensional']
        self.method = self.strings['level_set_method']
        self.ls_scheme = self.strings['ls_temporal_scheme']
        self.lsr_method = self.strings['ls_rein_option']
        self.element = self.strings['ls_fe_element']
        self.ps_element = self.strings['ps_element']
        self.element_orientation = self.strings['element_orientation']
        self.location = self.strings['comp_location']
        self.dimension = self.strings['dimension']
        self.mesh_from_file = self.strings['mesh_from_file']
        self.mesh_file_path = self.strings['mesh_file_path']
        self.write_bool = self.strings['write_to_file']
        self.extra = self.strings['extra']
        self.constitutive_equation = self.strings['constitutive_equation']
        self.constitutive_type = self.strings['constitutive_type']
        self.stability = self.strings['viscoelastic_stabilisation']
        self.axisymmetry = self.strings['axisymmetry']
        self.boundary_conditions = self.strings['boundary_conditions']
        self.DEVSSG = self.strings['DEVSSG']

        self.rein_steps = int(self.parameters['ls_rein_steps'])
        self.rein_div = int(self.parameters['ls_rein_divisor'])
        self.q_div = int(self.parameters['q_div'])
        self.ls_order = int(self.parameters['ls_fe_order'])
        self.T = self.parameters['T']
        self.num_steps = self.parameters['num_steps']
        self.dt = self.T/self.num_steps
        self.cox1 = self.parameters['cox1']#/0.6
        self.coy1 = self.parameters['coy1']#/0.6
        self.coz1 = self.parameters['coz1']
        self.cox2 = self.parameters['cox2']#/0.6
        self.coy2 = self.parameters['coy2']#/0.6
        self.coz2 = self.parameters['coz2']
        self.sizex = int(self.parameters['sizex'])
        self.sizey = int(self.parameters['sizey'])
        self.sizez = int(self.parameters['sizez'])
        self.centrex = self.parameters['centrex']#/0.6
        self.centrey = self.parameters['centrey']#/0.6
        self.centrez = self.parameters['centrey']
        self.radius = self.parameters['radius']#/0.6
        self.rho1 = self.parameters['rho_inner']
        self.rho2 = self.parameters['rho_outer']
        self.grav = self.parameters['gravity']
        self.d = self.parameters['d']

        if (self.fluid == 'Newtonian'):
            
            if (self.dimensional == 'Dim'):

                self.mu1 = self.parameters['mu_inner']
                self.mu2 = self.parameters['mu_outer']
                self.curvature = self.parameters['surface_tension_coefficient']

            elif (self.dimensional == 'NonDim'):
                
                self.mu1 = self.parameters['mu_inner']
                self.mu2 = self.parameters['mu_outer']
                self.Re1 = self.parameters['Re_inner']
                self.Re2 = self.parameters['Re_outer']
                self.Eo = self.parameters['Eo']
                self.curvature = 1/self.Eo
                self.Fr = self.parameters['Fr']

        elif (self.fluid == 'Viscoelastic'):

            # if (self.fluid == 'FENE-P-MP'):

            if (self.dimensional == 'Dim'):

                self.eta_s_in = self.parameters['etas_inner']
                self.eta_s_out = self.parameters['etas_outer']
                self.eta_p_in = self.parameters['etap_inner']
                self.eta_p_out = self.parameters['etap_outer']
                self.lamb_in = self.parameters['relaxation_time_in']
                self.lamb_out = self.parameters['relaxation_time_out']
                self.gmf = self.parameters['giesekus_mobility_factor']
                self.curvature = self.parameters['surface_tension_coefficient']
                self.b_fenepmp = self.parameters['b_fenepmp']
                self.lamb_fenepmp = self.parameters['lamb_fenepmp']

            elif (self.dimensional == 'NonDim'):

                self.cox1 = self.parameters['cox1']/0.00458
                self.coy1 = self.parameters['coy1']/0.00458
                self.cox2 = self.parameters['cox2']/0.00458
                self.coy2 = self.parameters['coy2']/0.00458
                self.centrex = self.parameters['centrex']/0.00458
                self.centrey = self.parameters['centrey']/0.00458
                self.radius = self.parameters['radius']/0.00458

                self.beta1 = self.parameters['beta_inner']
                self.beta2 = self.parameters['beta_outer']
                self.Re2 = self.parameters['Re_outer']
                self.Re_eps = self.parameters['Re_ratio']
                self.Wi1 = self.parameters['Wi_inner']
                self.Wi2 = self.parameters['Wi_outer']
                self.Eo = self.parameters['Eo']
                self.curvature = 1/self.Eo
                self.Fr = self.parameters['Fr']
                self.theta_in = self.parameters['theta_in']
                self.theta_out = self.parameters['theta_out']
                self.gmf = self.parameters['giesekus_mobility_factor']
                self.dt = (self.T/self.num_steps)*(np.sqrt(981*0.00458))/0.00458
                self.T = self.parameters['T']*(np.sqrt(981*0.00458))/0.00458
               
        if (self.rank == 0):

            for pair in self.parameters.items():

                print(pair)

            for pair in self.strings.items():

                print(pair)

    def read_parameters(self, parameters_file, strings_file):

        self.read_in(parameters_file, strings_file)

    # def read_parameters(self):

    #     if (self.rank == 0):

    #         print('Hello. Welcome to my rising bubble solver.\nWhat type of fluid are you modelling the bubble in?\n')
    #         print(f'[{bcolors.OKGREEN}0{bcolors.ENDC}]: Newtonian')
    #         print(f'[{bcolors.OKGREEN}1{bcolors.ENDC}]: Viscoelastic\n')

    #         self.fluid_choice = int(input())

    #         print('\nWould you like a dimensional or non-dimensional simulation?\n')
    #         print(f'[{bcolors.OKGREEN}0{bcolors.ENDC}]: Dimensional')
    #         print(f'[{bcolors.OKGREEN}1{bcolors.ENDC}]: Non-Dimensional\n')

    #         self.dimension_choice = int(input())

    #         if (self.dimension_choice == 1):

    #             print('\nDo your spatial / temporal variables need to be non-dimensionalised?\n')
    #             print(f'[{bcolors.OKGREEN}0{bcolors.ENDC}]: Yes')
    #             print(f'[{bcolors.OKGREEN}1{bcolors.ENDC}]: No\n')

    #             self.dimension_convert_choice = int(input())

    #         else:

    #             self.dimension_convert_choice = None

    #         if (self.fluid_choice == 0):

    #             if (self.dimension_choice == 0):

    #                 print(f"\n{bcolors.OKCYAN}Newtonian (Dimensional) parameters:{bcolors.ENDC}")

    #             elif (self.dimension_choice == 1):

    #                 print(f"\n{bcolors.OKCYAN}Newtonian (Non-Dimensional) parameters:{bcolors.ENDC}")

    #         elif (self.fluid_choice == 1):

    #             if (self.dimension_choice == 0):

    #                 print(f"\n{bcolors.OKCYAN}Viscoelastic (Dimensional) parameters:{bcolors.ENDC}")

    #             elif (self.dimension_choice == 1):

    #                 print(f"\n{bcolors.OKCYAN}Viscoelastic (Non-Dimensional) parameters:{bcolors.ENDC}")

    #     else:

    #         self.fluid_choice = None
    #         self.dimension_choice = None
    #         self.dimension_convert_choice = None

    #     self.fluid_choice = self.comm.bcast(self.fluid_choice, root = 0)
    #     self.dimension_choice = self.comm.bcast(self.dimension_choice, root = 0)
    #     self.dimension_convert_choice = self.comm.bcast(self.dimension_convert_choice, root = 0)

    #     if (self.fluid_choice == 0):

    #         self.fluid = 'Newtonian'

    #         if (self.dimension_choice == 0):

    #             self.dimensional = 'Dim'
    #             self.read_in('Newtonian_D.txt')

    #         elif (self.dimension_choice == 1):

    #             self.dimensional = 'NonDim'
    #             self.read_in('Newtonian_ND.txt')

    #     elif (self.fluid_choice == 1):

    #         self.fluid = 'Viscoelastic'

    #         if (self.dimension_choice == 0):

    #             self.dimensional = 'Dim'
    #             self.read_in('Viscoelastic_D.txt')

    #         elif (self.dimension_choice == 1):

    #             self.dimensional = 'NonDim'
    #             self.read_in('Viscoelastic_ND.txt')

        # if (self.rank == 0):

        #     os.rmdir(f'{self.method}_{self.fluid}_{self.element}_{self.dimensional}')
        #     os.mkdir(f'{self.method}_{self.fluid}_{self.element}_{self.dimensional}')

                # if (self.dimension_convert_choice == 0):

        # self.char_length = self.radius
        # self.char_vel = np.sqrt(self.char_length*self.grav)

        # self.T = self.T*(self.char_vel/self.char_length)
        # self.dt = self.dt*(self.char_vel/self.char_length)
        # self.cox1 = self.cox1/self.char_length
        # self.coy1 = self.coy1/self.char_length
        # self.cox2 = round(self.cox2/self.char_length,5)
        # self.coy2 = round(self.coy2/self.char_length,5)
        # self.centrex = self.centrex/self.char_length
        # self.centrey = self.centrey/self.char_length
        # self.radius = self.radius/self.char_length
        # self.grav = 1

        # elif (self.dimension_convert_choice == 1 or self.dimension_choice == 0):

        #     self.char_length = 1
        #     self.char_vel = 1