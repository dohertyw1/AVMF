from fenics import *
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from Funcs import *

class Process():

    def __init__(self):

        set_log_active(False)

    """Load in data from the Multiphase.py solver."""
    def load_data(self, suffix):

        with open(f'{self.file_string}/{self.file_string}_{suffix}.csv', newline='') as csvfile:

            self.raw_data = np.array(list(csv.reader(csvfile, delimiter='\t')))

    """Sort raw data into respective lists of data."""
    def categorise_data(self):

        self.timescale = []
        self.area = []
        self.xcom = []
        self.ycom = []
        self.circ = []
        self.urise = []
        self.vrise = []

        variables = [self.timescale, self.area, self.xcom, self.ycom, self.circ, self.urise, self.vrise]

        for variable in variables:

            for i in range(np.shape(self.raw_data)[0]):

                variable.append(float(self.raw_data[i][variables.index(variable)]))

    """Load in the Newtonian benchmark data from bubble_benchmarks file."""
    def load_benchmark_data(self, case):

        raw_FreeLIFE_timescale = []
        raw_FreeLIFE_area  = []
        raw_FreeLIFE_circ  = []
        raw_FreeLIFE_ycom  = []
        raw_FreeLIFE_vrise  = []

        raw_FreeLIFE = [raw_FreeLIFE_timescale, raw_FreeLIFE_area, raw_FreeLIFE_circ, raw_FreeLIFE_ycom, raw_FreeLIFE_vrise]

        self.FreeLIFE_timescale = []
        self.FreeLIFE_area  = []
        self.FreeLIFE_circ  = []
        self.FreeLIFE_ycom  = []
        self.FreeLIFE_vrise  = []

        self.FreeLIFE = [self.FreeLIFE_timescale, self.FreeLIFE_area, self.FreeLIFE_circ, self.FreeLIFE_ycom, self.FreeLIFE_vrise]

        for variable in raw_FreeLIFE:

            with open(f'bubble_benchmarks/FreeLIFE_{case}.txt') as inf:

                reader = csv.reader(inf, delimiter=" ")

                variable.append(list(zip(*reader))[raw_FreeLIFE.index(variable)+1])

        for variable in self.FreeLIFE:

            for i in range(np.size(raw_FreeLIFE[self.FreeLIFE.index(variable)])):

                variable.append(float(raw_FreeLIFE[self.FreeLIFE.index(variable)][0][i]))

        raw_MooNMD_timescale = []
        raw_MooNMD_area  = []
        raw_MooNMD_circ  = []
        raw_MooNMD_ycom  = []
        raw_MooNMD_vrise  = []

        raw_MooNMD = [raw_MooNMD_timescale, raw_MooNMD_area, raw_MooNMD_circ, raw_MooNMD_ycom, raw_MooNMD_vrise]

        self.MooNMD_timescale = []
        self.MooNMD_area  = []
        self.MooNMD_circ  = []
        self.MooNMD_ycom  = []
        self.MooNMD_vrise  = []

        self.MooNMD = [self.MooNMD_timescale, self.MooNMD_area, self.MooNMD_circ, self.MooNMD_ycom, self.MooNMD_vrise]

        for variable in raw_MooNMD:

            with open(f'bubble_benchmarks/MooNMD_{case}.txt') as inf:

                reader = csv.reader(inf, delimiter=" ")

                variable.append(list(zip(*reader))[raw_MooNMD.index(variable)+1])

        for variable in self.MooNMD:

            for i in range(np.size(raw_MooNMD[self.MooNMD.index(variable)])):

                variable.append(float(raw_MooNMD[self.MooNMD.index(variable)][0][i]))

    """Load in the saved bubble shape from the last temporal iteration."""
    def load_bubble_shape(self, mesh_saved):

        if (mesh_saved):

            mesh = Mesh()

            with XDMFFile(f'{self.file_string}/phi_read.xdmf') as infile:

                infile.read(mesh)

        else:

            mesh = RectangleMesh(Point(0,0), Point(self.height, self.length), self.nx, self.ny)

        V = FunctionSpace(mesh, 'CG', self.ls_order) 
        self.level_set = Function(V)

        with XDMFFile(f'{self.file_string}/phi_read.xdmf') as infile:

            infile.read_checkpoint(self.level_set, "phi")

    """Plot for various quantities."""
    def individual_plotter(self, variable):

        if (variable == 'Drag'):

            current_var = self.area
            MooNMD_var = self.MooNMD_area
            FreeLIFE_var = self.FreeLIFE_area

        elif (variable == 'Circularity'):

            current_var = self.circ
            MooNMD_var = self.MooNMD_circ
            FreeLIFE_var = self.FreeLIFE_circ

        elif (variable == 'Centre of mass'):

            current_var = self.ycom
            MooNMD_var = self.MooNMD_ycom
            FreeLIFE_var = self.FreeLIFE_ycom

        elif (variable == 'Rise Velocity'):

            current_var = self.vrise
            MooNMD_var = self.MooNMD_vrise
            FreeLIFE_var = self.FreeLIFE_vrise

        plt.plot(self.timescale, current_var, color='black')

        if (self.fluid == 'Newtonian'):

            plt.plot(self.MooNMD_timescale, MooNMD_var,color='red',linestyle=':')
            plt.plot(self.FreeLIFE_timescale,FreeLIFE_var,color='blue',linestyle='-.')
            plt.xlim([0,3])
            
        plt.grid()
        plt.title(f'{variable}')
        plt.legend(['Current study', 'MooNMD Benchmark','FreeLIFE Benchmark'])
        plt.tight_layout()

        """Calculate benchmark data."""
    def process_data(self):

        if (self.method == 'NCons' and self.dimension == '2D'):

            area = assemble(conditional(lt(self.phi0, 0), 1.0, 0.0)*dx)
            x_com = assemble(Expression("x[0]", degree = 1)*(conditional(lt(self.phi0, 0), 1.0, 0.0))*dx)/area
            y_com = assemble(Expression("x[1]", degree = 1)*(conditional(lt(self.phi0, 0), 1.0, 0.0))*dx)/area

            Pa = 2.0*sqrt(np.pi*area)
            Pb = assemble(mgrad(self.phi0)*delta_func(self.phi0/sqrt(dot(grad(self.phi0),grad(self.phi0))), self.eps)*dx)
            circ = Pa/Pb

            u_rise = assemble(self.u0[0]*(conditional(lt(self.phi0, 0), 1.0, 0.0))*dx)/area
            v_rise = assemble(self.u0[1]*(conditional(lt(self.phi0, 0), 1.0, 0.0))*dx)/area

        if (self.method == 'Cons' and self.dimension == '2D'):

            area = assemble(self.phi0*self.x_axi[0]*dx)
            area_x2 = 2*area

            x_com = assemble(Expression("x[0]", degree = 1)*self.phi0*self.x_axi[0]*dx)/area
            y_com = assemble(Expression("x[1]", degree = 1)*self.phi0*self.x_axi[0]*dx)/area

            Pa = 2.0*sqrt(np.pi*area_x2)
            Pb = 2*assemble(mgrad(self.phi0)*self.x_axi[0]*dx)
            circ = Pa/Pb

            u_rise = assemble(self.u0[0]*self.phi0*self.x_axi[0]*dx)/area
            v_rise = assemble(self.u0[1]*self.phi0*self.x_axi[0]*dx)/area

            # force_x = (-self.p0+self.tau0[0,0]+2*self.beta*self.u0[0].dx(0))*ngamma(self.phi0)[0]
            # force_y = (self.tau0[1,0]+self.beta*(self.u0[0].dx(1)+self.u0[1].dx(0)))*ngamma(self.phi0)[1]

            # drag = -assemble(2*(force_x+force_y)*self.phi0*dx)

        self.timeseries = [self.t, area_x2, x_com, y_com, circ, u_rise, v_rise]

        if self.rank == 0:
            with open((f"{self.file_string}/{self.file_string}_data.csv"), 'a') as csvfile:
                f = csv.writer(csvfile, delimiter='\t',lineterminator='\n',)
                f.writerow(self.timeseries)

    """Write final level-set function to separate file in order to retrieve the bubble shape"""
    def process_shape(self, frame):

        if (self.method == 'NCons'):

            with XDMFFile(f"{self.file_string}/phi_read.xdmf") as outfile:

                outfile.write_checkpoint(self.phi0, "phi", 0, append=True)

        elif (self.method == 'Cons'):

            with XDMFFile(f"{self.file_string}/frames/frame_{np.round(frame, 4)}/phi_read_{np.round(frame, 4)}.xdmf") as outfile:

                outfile.write_checkpoint(self.phi0, "phi", 0, append=True)

            with XDMFFile(f"{self.file_string}/frames/frame_{np.round(frame, 4)}/u_read_{np.round(frame, 4)}.xdmf") as outfile:

                outfile.write_checkpoint(self.u0, "u", 0, append=True)
            
            with XDMFFile(f"{self.file_string}/frames/frame_{np.round(frame, 4)}/p_read_{np.round(frame, 4)}.xdmf") as outfile:

                outfile.write_checkpoint(self.p0, "p", 0, append=True)

            with XDMFFile(f"{self.file_string}/frames/frame_{np.round(frame, 4)}/tau_read_{np.round(frame, 4)}.xdmf") as outfile:

                outfile.write_checkpoint(self.tau0, "tau", 0, append=True)

    """Set up files to save data and delete old data."""
    def set_up_files(self):

        self.xdmf_file_phi = XDMFFile(f'{self.file_string}/phi.xdmf')
        self.xdmf_file_u = XDMFFile(f'{self.file_string}/u.xdmf')
        self.xdmf_file_p = XDMFFile(f'{self.file_string}/p.xdmf')

        if (self.fluid == 'Viscoelastic'):

            self.xdmf_file_tau = XDMFFile(f'{self.file_string}/tau.xdmf')
            self.xdmf_file_tau.parameters['flush_output'] = True 

        if (self.rank == 0 and os.path.isfile(f'{self.file_string}/{self.file_string}_data.csv') == True):

            os.remove(f'{self.file_string}/{self.file_string}_data.csv')
          
        self.xdmf_file_phi.parameters['flush_output'] = True
        self.xdmf_file_u.parameters['flush_output'] = True
        self.xdmf_file_p.parameters['flush_output'] = True  

    """Write the solution to file."""
    def write_to_file(self):

        self.xdmf_file_phi.write(self.phi0, self.t)
        self.xdmf_file_u.write(self.u0, self.t)
        # self.xdmf_file_p.write(self.p0, self.t)

        if (self.fluid == 'Viscoelastic'):
            
            self.xdmf_file_tau.write(self.tau0, self.t)

    def write_parameters(self):

        if (self.rank == 0):

            p_file = open(f"{self.file_string}/parameters.txt", "w")

            for pair in self.parameters.items():

                p_file.write(f"{pair}\n")

            for pair in self.strings.items():

                p_file.write(f"{pair}\n")

            # p_file.write(f"('eps', 980.0)")

            p_file.close()

            # print(f"{bcolors.OKCYAN}\nRunning the simulation...\n{bcolors.ENDC}")

    def sns_drag(self,p,u,tau):

        force_x = (-p+tau[0,0]+2*self.beta*u[0].dx(0))*self.nn[0]
        force_y = (tau[1,0]+self.beta*(u[0].dx(1)+u[1].dx(0)))*self.nn[1]
        self.drag_coeff = -assemble(2*(force_x+force_y)*self.ds_circle)

    def sns_drag_to_file(self, drag_file):

        if self.rank == 0:
            with open((f"{drag_file}_{self.We}.csv"), 'a') as csvfile:
                f = csv.writer(csvfile, delimiter='\t',lineterminator='\n',)
                f.writerow([self.drag_coeff])


    """Run the post processing."""
    def post_process(self):

        self.load_data('data')

        self.categorise_data()

        self.load_benchmark_data('case1')

        # self.load_bubble_shape('Cons', 'Newtonian')

        if (self.rank == 0):

            plt.figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')

            plt.subplot(221)
            self.individual_plotter('Drag')
            plt.subplot(222)
            self.individual_plotter('Circularity')
            plt.subplot(223)
            self.individual_plotter('Centre of mass')
            plt.subplot(224)
            self.individual_plotter('Rise Velocity')
            plt.suptitle(f'{self.method}_{self.fluid}_{self.element}')
            plt.savefig(f'{self.file_string}/test1{self.element}')
            # plt.show()

            plt.figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')

