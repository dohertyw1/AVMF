import matplotlib.pyplot as plt
from fenics import *
import numpy as np
from tqdm import tqdm
import math
import ufl as uf
import csv
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

set_log_active(False)

parameters["std_out_all_processes"] = False
parameters['ghost_mode'] = 'shared_facet'
parameters["mesh_partitioner"] = "ParMETIS"

file_phi = XDMFFile('postproc/phi.xdmf')
file_u = XDMFFile('postproc/u.xdmf')
file_p = XDMFFile('postproc/p.xdmf')
file_tau = XDMFFile('postproc/tau.xdmf')

file_phi.parameters['flush_output'] = True
file_u.parameters['flush_output'] = True
file_p.parameters['flush_output'] = True
file_tau.parameters['flush_output'] = True

rho_in = 0.1

beta_in = 1

rho_out = 1
Re_in = 14.19436759
Re_out = 1.419436759
De_in = 0
De_out = 8.082903769
We_out = 35.28
beta_out = 0.07

nx = 100
ny = 2*nx

mesh = RectangleMesh(Point(0, 0), Point(3.33333, 6.66666), nx, ny, 'crossed')

eps = Constant(1.5*mesh.hmin()/4)


def rho(phi):

    return rho_in*phi + rho_out*(1-phi)


def De(phi):

    return De_in*phi + De_out*(1-phi)


def We(phi):

    return We_out


def Re(phi):

    return Re_in*phi + Re_out*(1-phi)


def beta(phi):

    return beta_in*phi + beta_out*(1-phi)


def magnitude(u):
    return sqrt(u**2)


def epsilon(u):
    return sym(grad(u))


def trans(u):
    return u.T


def mgrad(phi):
    return sqrt(dot(grad(phi), grad(phi)))


def ngamma(phi):
    return grad(phi)/sqrt(dot(grad(phi), grad(phi)))


def CDelta(phi, eps):

    return conditional(lt(abs(phi), eps), 1.0/(2.0*eps)*(1.0 + uf.cos(np.pi*phi/eps)), 0.0)


def curvature(phi, sig, v):

    return (1/We(phi))*mgrad(phi)*inner((Identity(2) - outer(ngamma(phi), ngamma(phi))), epsilon(v))*dx


def CHeaviside(phi, eps):

    return conditional(lt(abs(phi), eps), 0.5*(1.0 + phi/eps + 1/np.pi*uf.sin(np.pi*phi/eps)), (uf.sign(phi) + 1)/2.0)


T = 0.3*(np.sqrt(980*0.6))/0.6  # 5.25


num_steps_rein = 3
rein_div = 1


dt = (0.0005*(np.sqrt(980*0.6))/0.6)  # 0.0404
num_steps = int(T/dt)
g = Constant((0, 1))

scaling = 0.1
toler = 1.0e-3


dist = Expression('sqrt( (pow((x[0]-A),2)) + (pow((x[1]-B),2)) )-r',
                  degree=2, A=1/0.6, B=1/0.6, r=0.5)


dtau = Constant(0.5*mesh.hmin()**1.10)

dist2 = Expression('(1/(1+exp((dist/eps))))',
                   degree=2, eps=eps, dist=dist)

""" LEVEL SET """
Q = FiniteElement('CG', mesh.ufl_cell(), 2)
Qs = FunctionSpace(mesh, Q)

phi = TrialFunction(Qs)
psi = TestFunction(Qs)

phi00 = Function(Qs)
phi0 = interpolate(dist2, Qs)

phi_rein = Function(Qs)

""" LEVEL SET REIN """
Vnormal = VectorFunctionSpace(mesh, 'CG', 1)

phigrad = Function(Vnormal)
vnorm = TestFunction(Vnormal)

""" PRESSURE """
P = FiniteElement('CG', mesh.ufl_cell(), 1)
Ps = FunctionSpace(mesh, P)

p = TrialFunction(Ps)
q = TestFunction(Ps)

p0 = Function(Ps)
p_ = Function(Ps)

""" VELOCITY """
V = VectorElement('CG', mesh.ufl_cell(), 2)
Vs = FunctionSpace(mesh, V)

u = TrialFunction(Vs)
v = TestFunction(Vs)

u0 = Function(Vs)
u_ = Function(Vs)

""" STRESS """
T = TensorElement('CG', mesh.ufl_cell(), 2)
Ts = FunctionSpace(mesh, T)

tau = TrialFunction(Ts)
S = TestFunction(Ts)

tau0 = Function(Ts)

walls = 'near(x[1], 0) || near(x[1], 6.66666)'
fswalls = 'near(x[0], 0) || near(x[0], 3.33333)'

bcu_noslip = DirichletBC(Vs, Constant((0, 0)), walls)
bcu_fslip = DirichletBC(Vs.sub(0), Constant(0), fswalls)

n = FacetNormal(mesh)

bcu = [bcu_noslip, bcu_fslip]

F_levelset = (phi/dt)*psi*dx - (phi0/dt)*psi*dx + inner(u0, grad(phi))*psi*dx \
    + ((phi - phi0)/dt + inner(u0, grad(phi))) \
    * scaling*mesh.hmin()/uf.Max(2.0*sqrt(inner(u0, u0)), toler/mesh.hmin())*inner(u0, grad(psi))*dx

F_rein = (phi_rein - phi0)/dtau*psi*dx \
    - phi_rein*(1.0 - phi_rein)*inner(grad(psi), phigrad)*dx \
    + eps*inner(grad(phi_rein), grad(psi))*dx

F_grad = inner((phigrad-ngamma(phi0)), vnorm)*dx

phi = Function(Qs)

alpha = 0.1

F_stress = (1/dt)*inner(De(phi)*(tau-tau0), S)*dx \
    + 0.5*((inner(dot(De(phi)*u0, nabla_grad(tau))
                  - dot(De(phi)*tau, trans(grad(u0)))
                  - dot(De(phi)*grad(u0), tau), S)*dx
            + inner(tau, S)*dx)
           - (1-beta(phi))*inner(grad(u0)+trans(grad(u0)), S)*dx) \
    + 0.5*((inner(dot(De(phi)*u0, nabla_grad(tau0))
                  - dot(De(phi)*tau0, trans(grad(u0)))
                  - dot(De(phi)*grad(u0), tau0), S)*dx
            + inner(tau0, S)*dx)
           - (1-beta(phi))*inner(grad(u0)+trans(grad(u0)), S)*dx) \
    + inner(alpha*(De(phi)/(1-beta(phi)))*tau0*tau0, S)*dx \

ns1 = (1/dt)*inner((rho(phi)*u - rho(phi00)*u0), v)*dx \
    + inner(dot(rho(phi)*u0, nabla_grad(u)), v)*dx \
    + (beta(phi)/Re(phi))*inner(grad(u), grad(v))*dx \
    - p0*div(v)*dx \
    + inner(rho(phi)*g, v)*dx \
    + curvature(phi, None, v) \
    + (1/Re_out)*inner(tau0, grad(v))*dx \

ns2 = (1/rho(phi))*(dot(grad(p), grad(q)) - dot(grad(p0), grad(q)))*dx \
    + (1/dt)*div(u_)*q*dx \

ns3 = inner(u, v)*dx - inner(u_, v)*dx \
    + (dt/rho(phi))*inner(grad(p_-p0), v)*dx

bcs = []

tol = 1.0e-4

tau = Function(Ts)

t = 0
q = 0

for n in tqdm(range(num_steps)):

    t += dt
    q += 1

    a = assemble(lhs(F_levelset))
    L = assemble(rhs(F_levelset))

    solve(a, phi.vector(), L, 'gmres', 'default')

    if n % rein_div == 0 and n > 0:

        phi0.assign(phi)

        solve(F_grad == 0, phigrad)

        for n in range(num_steps_rein):

            solve(F_rein == 0, phi_rein, solver_parameters={"newton_solver": {"linear_solver": 'gmres', "preconditioner": 'default',
                                                                              "maximum_iterations": 20, "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}},
                  form_compiler_parameters={"optimize": True})

            phi00.assign(phi0)
            phi0.assign(phi_rein)

        phi.assign(phi_rein)

    else:

        phi00.assign(phi0)

    A = assemble(lhs(F_stress))
    M = assemble(rhs(F_stress))

    solve(A, tau.vector(), M, 'gmres', 'default')

    A1 = assemble(lhs(ns1))
    M1 = assemble(rhs(ns1))
    [bc.apply(A1) for bc in bcu]
    [bc.apply(M1) for bc in bcu]
    solve(A1, u_.vector(), M1, 'gmres', 'default')

    M2 = assemble(rhs(ns2))
    A2 = assemble(lhs(ns2))
    solve(A2, p_.vector(), M2, 'gmres', 'default')

    A3 = assemble(lhs(ns3))
    M3 = assemble(rhs(ns3))
    [bc.apply(A3) for bc in bcu]
    [bc.apply(M3) for bc in bcu]
    solve(A3, u_.vector(), M3, 'gmres', 'default')

    if q % 12 == 0:

        file_phi.write(phi, t)

    u0.assign(u_)
    p0.assign(p_)
    tau0.assign(tau)
    phi0.assign(phi)
