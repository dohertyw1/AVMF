from fenics import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import meshio
from mpi4py import MPI
import csv

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
set_log_active(False)

parameters["std_out_all_processes"] = False;
parameters['ghost_mode'] = 'shared_facet' 
parameters["mesh_partitioner"] = "ParMETIS"
# parameters['krylov_solver']['nonzero_initial_guess'] = True

beta = 0.59
theta = 0.41
mesh = Mesh('godzilla.xml')

drag_list = []
nn = FacetNormal(mesh)

def epsilon(a):
    return sym(grad(a))
def trans(a):
    return a.T
def magnitude(vec):
    return sqrt(vec**2)
def flux(ui, ni):
    return (dot(ui, ni) + abs(dot(ui, ni)))/2
def drag_func(p, u, tau):
    force_x = (-p+tau[0,0]+2*beta*u[0].dx(0))*n[0]
    force_y = (tau[1,0]+beta*(u[0].dx(1)+u[1].dx(0)))*n[1]
    drag_coeff = -assemble(2*(force_x+force_y)*ds_circle)
    return drag_coeff

P = FiniteElement('CG', mesh.ufl_cell(), 1)
V = VectorElement('CG', mesh.ufl_cell(), 2)
S = TensorElement('CG', mesh.ufl_cell(), 2)
D = TensorElement('DG', mesh.ufl_cell(), 0)

Pr = FunctionSpace(mesh, P)
Ve = FunctionSpace(mesh, V)
St = FunctionSpace(mesh, S)
De = FunctionSpace(mesh, D)

stokes = FunctionSpace(mesh, MixedElement([V, P]))
n = FacetNormal(mesh)

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
        # Cylinder
        return on_boundary and (between(x[0], (-1, 1)) and between(x[1], (0, 1)))
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
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
ds_circle = Measure("ds", subdomain_data=subdomains, subdomain_id=5)
ds_out = Measure("ds", subdomain_data=subdomains, subdomain_id=4)

# file_u = XDMFFile('singlephaseOB/u.xdmf')
# file_u.parameters['flush_output'] = True

# file_tau = XDMFFile('singlephaseOB/tau.xdmf')
# file_tau.parameters['flush_output'] = True

# file_p = XDMFFile('singlephaseOB/p.xdmf')
# file_p.parameters['flush_output'] = True

tau = TrialFunction(St)
S = TestFunction(St)
tau1 = Function(St)

u, p= TrialFunctions(stokes)
v, q = TestFunctions(stokes)

w1 = Function(stokes)
u1, p1 = split(w1)

w11 = Function(stokes)
u11, p11, = split(w11)

ws = Function(stokes)
us, ps = split(ws)

taus = Function(St)

G = TrialFunction(De)
Gs = TestFunction(De)
G1=Function(De)

def bcs(We):

    vel_inflow = Expression(('1.5*(1-(x[1]*x[1]/4))', '0'), degree=2)

    str_inflow = Expression((('2*We*(1-beta)*(9/16)*x[1]*x[1]',
                                '-3/4*(1-beta)*x[1]'),
                                ('-3/4*(1-beta)*x[1]', '0')),
                                degree=2, We=We, beta=beta)

    stress_inflow_0 = DirichletBC(St, str_inflow, subdomains, 1)
    velocity_inflow_1 = DirichletBC(stokes.sub(0), vel_inflow, subdomains, 1)
    noslip_top_1 = DirichletBC(stokes.sub(0), vel_inflow, subdomains, 2)
    symmetry_1 = DirichletBC(stokes.sub(0).sub(1), Constant(0), subdomains, 3)
    outflow_condition_1 = DirichletBC(stokes.sub(0).sub(1), Constant(0), subdomains, 4)
    outflow_pressure_1 = DirichletBC(stokes.sub(1), Constant(0), subdomains, 4)
    noslip_wedge_1 = DirichletBC(stokes.sub(0), Constant((0, 0)), subdomains, 5)

    bcst = [stress_inflow_0]

    bcup = [velocity_inflow_1,
            noslip_top_1,
            symmetry_1,
            outflow_condition_1,
            noslip_wedge_1,
            outflow_pressure_1]
    
    bcup2 = [noslip_top_1,
            symmetry_1,
            outflow_condition_1,
            noslip_wedge_1,
            outflow_pressure_1]

    return bcst, bcup, bcup2

def first_step(We, T, num_steps, ws, taus):


    u1, p1 = split(w1)

    vgp = inner(G-grad(u1),Gs)*dx


    bcst, bcup, bcup2 = bcs(We)

    dt = T / num_steps

    h = CellDiameter(mesh)
    supg = (h/magnitude(u1)+0.000001)*dot(u1,nabla_grad(S))

        
    stokes_system = Constant(2.0)*(beta+theta)*inner(epsilon(u),epsilon(v))*dx \
                - theta*inner(G1+trans(G1),epsilon(v))*dx \
                - p*div(v)*dx + q*div(u)*dx \
                + inner(tau1,grad(v))*dx    

    stress_cn = inner(tau, S)*dx \
        - (1-beta)*inner(G1+trans(G1), S)*dx \
        + (We/dt)*inner(tau-taus, S)*dx \
        + We*inner(dot(u1, nabla_grad(tau))
        - dot(tau, trans(G1))
        - dot(G1, tau), S)*dx \
        + We*inner(dot(u1, nabla_grad(tau)), supg)*dx \
        

    t = 0
    qt=0

    for n in tqdm(range(num_steps)): 

        qt+=1

        stokes_l = assemble(lhs(stokes_system))
        stokes_r = assemble(rhs(stokes_system))

        [bc.apply(stokes_l, stokes_r) for bc in bcup]

        solve(stokes_l, w1.vector(), stokes_r )

        u1, p1 = split(w1)

        vgp_l = assemble(lhs(vgp))
        vgp_r = assemble(rhs(vgp))

        solve(vgp_l, G1.vector(), vgp_r, 'bicgstab', 'default')

        stress_cn_l = assemble(lhs(stress_cn))
        stress_cn_r = assemble(rhs(stress_cn)) 

        [bc.apply(stress_cn_l, stress_cn_r) for bc in bcst]

        solve(stress_cn_l, tau1.vector(), stress_cn_r, 'bicgstab', 'default')

        # solve(stress_cn == 0, tau, bcs=bcst, solver_parameters={"newton_solver": {"linear_solver": 'gmres', \
        #                                        "preconditioner": 'default', "maximum_iterations": 20, \
        #                                        "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
        #                                        form_compiler_parameters={"optimize": True})

        drag_coeff = drag_func(p1, u1, tau1)

        if rank == 0:
            with open((f"GODZILLA_SUPG_{We}.csv"), 'a') as csvfile:
                f = csv.writer(csvfile, delimiter='\t',lineterminator='\n',)
                f.writerow([drag_coeff])
                    
        taus.assign(tau1)
        # ws.assign(w1)

    return ws, taus

for i, s in zip([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                [3,3,3,5,5,5,5,5,5,5]):

    ws, taus = first_step(i, s, int(s*100), ws, taus)
    # us,ps=split(ws)
    # np.array(drag_list).tofile('cg'+str(i))
    # drag_list=[]

