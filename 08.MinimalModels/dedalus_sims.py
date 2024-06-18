import numpy as np
import dedalus.public as d3
import logging
import os
from scipy.interpolate import RectBivariateSpline
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

def run_and_save(params, filename, logger, m0, c0):
    Lx, Ly = params["Lx"], params["Ly"]
    Nx, Ny = params["Nx"], params["Ny"]
    dealias = 3/2
    
    mu = params["mu"]
    α  = params["α"]
    τ  = params.get("τ", 1e5)
    A  = params.get("A", 0)
    B  = params.get("B", 0)
    C  = params.get("C", 0)
    D  = params.get("D", 0)
    km = params.get("km", 0)
    kE = params.get("kE", 0)

    # Bases
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(-Lx/2,Lx/2), dealias=dealias)
    ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
    x, y = dist.local_grids(xbasis, ybasis)

    # Fields
    c = dist.ScalarField(name='c', bases=(xbasis,ybasis))
    v = dist.VectorField(coords, name="v", bases=(xbasis,ybasis))
    m = dist.TensorField(coords, name="m", bases=(xbasis, ybasis))
    p = dist.Field(name='p', bases=(xbasis,ybasis))
    
    #Fixing gauge conditions
    tau_p = dist.Field(name='tau_p')
    tau_v = dist.VectorField(coords, name='tau_u')

    #Substitutions
    Ω = (d3.grad(v) - d3.transpose(d3.grad(v))) / 2
    E = (d3.grad(v) + d3.transpose(d3.grad(v))) / 2

    #Static velocity Problem
    problem = d3.IVP([c, m, v, p, tau_p, tau_v], namespace=locals())
    problem.add_equation("grad(p) - mu*lap(v) - α * div(m) + tau_v = 0") # Stokes equation (with velocity tau term)
    problem.add_equation("div(v) + tau_p = 0")  #Pressure tau term for stokes equation
    problem.add_equation("integ(p) = 0")        #Fix pressure gauge
    problem.add_equation("integ(v) = 0")        #Fix velocity gauge
    
    #Protein fields
    problem.add_equation("dt(c) = - v@grad(c)") #Cadherin advection
    eq_m  = "dt(m) - D*lap(m) ="
    eq_m += " -(Ω @ m - m @ Ω)  - v @ grad(m)" #Advection
    eq_m += " + (1 + km * c) / τ * m" #Linear growth/decay
    eq_m += " + A*m*trace(m) + B * m * trace(m)**2" #Higher-order feedback for setting FPs
    eq_m += "  + C * (1 + kE * c) * (E @ m + m @ E)" #Mechanical feedback
    problem.add_equation(eq_m)

    #Myosin initial conditions
    x0 = np.linspace(-Lx/2, Lx/2, m0.shape[-1])
    y0 = np.linspace(-Ly/2, Ly/2, m0.shape[-2])
    m["g"][0,0,:,:] = RectBivariateSpline(x0, y0, m0[3].T)(x, y)
    m["g"][0,1,:,:] = RectBivariateSpline(x0, y0, m0[1].T)(x, y)
    m["g"][1,0,:,:] = m['g'][0,1,:,:]
    m["g"][1, 1,:,:] = RectBivariateSpline(x0, y0, m0[0].T)(x, y)

    #Cadherin initial conditions
    x0 = np.linspace(-Lx/2, Lx/2, c0.shape[-1])
    y0 = np.linspace(-Ly/2, Ly/2, c0.shape[-2])
    c['g'][:] = RectBivariateSpline(x0, y0, c0.T)(x, y)
    
    v["g"][:] = 0
    
    # Solver
    solver = problem.build_solver(d3.SBDF2)
    solver.stop_sim_time = params['max_time']

    os.makedirs(f'dedalus_runs/{filename}', exist_ok=True)
    analysis = solver.evaluator.add_file_handler(f'dedalus_runs/{filename}', 
                                                 iter=params['save_every'], 
                                                 mode='append')
    analysis.add_tasks(solver.state, layout='g')
    
    # Main loop
    try:
        logger.info("Starting main loop")
        while solver.proceed:
            solver.step(params['timestep'])
            if np.isnan(np.mean(v["g"])) or np.isnan(np.mean(m["g"])) or np.isnan(np.mean(c["g"])):
                logger.error("computation diverges: adjust dt and/or dx")
                raise Exception("computation diverges: adjust dt and/or dx")
            if (solver.iteration-1) % 100 == 0:
                logger.info(f"Iteration={solver.iteration}, Time={solver.sim_time}")
    except:
        logger.error("Exception raised, triggering end of main loop.")
    finally:
        #Write parameters to file
        with analysis.get_file() as h5f:
            prms = h5f.create_group('params')
            for key in params:
                prms[key] = params[key]

if __name__ == '__main__':
    initial_conditions = np.load('initial_conditions.npz')
    tt = 0
    t0 = initial_conditions['t0']
    m0 = initial_conditions['m0'][tt, ..., 20:-20] #Crop AP poles
    c0 = initial_conditions['c0'][tt, ..., 20:-20].squeeze()
    print(f'Initial time = {t0[tt]}')

    # eCadherin case
    base_params = dict(
        Lx=452., Ly=533.,   #Units of microns
        Nx=256, Ny=256,     #Not equal to check transpose
        max_time=50, timestep=0.1, save_every=5,
    )
    cases = [
        #['Cadherin', dict(mu = 1., α = 1., τ = 3, A = -1, km=-0.6)],
        #['Cadherin', dict(mu = 1., α = 3., τ = 3, A = -1, C=0.5, km=-0.6, kE=0.25)], #Mechanical feedback
        #['Cadherin', dict(mu = 1., α = 1., τ = 3, A = -1, C=0.5, km=-0.6, kE=0.25)], #Mechanical feedback
        #['Cadherin', dict(mu = 1., α = 1., τ = 3, A = -1, km=-0.6)],
        ['Cadherin', dict(mu = 1., α = 1., τ = 3, A = -0.66, km=-0.6)],

    ]
    for filename, params in cases:
        print(f'Running {filename}')
        params = {**base_params, **params}
        run_and_save(params, filename, logger, m0=m0, c0=c0)

    # Myosin-only cases
    base_params = dict(
        mu = 1., α = 3., #Stokes terms, roughly correct units
        Lx=452., Ly=533.,   #Units of microns
        Nx=64, Ny=64,     #Not equal to check transpose
        max_time=100, timestep=0.1, save_every=5,
    )

    cases = [
        #['Passive', {}],
        #['Linear', dict(τ = +50)], #Growth
        #['Linear', dict(τ = -50)], #Decay
        #['Quadratic', dict(τ = -20, A = +0.4)], #FP > m0, unstable
        #['Quadratic', dict(τ = +20, A = -0.4)], #FP > m0, stable
        ['Quadratic', dict(τ = -20, A = +1.0)], #FP < m0, unstable
        #['Quadratic', dict(τ = +20, A = -1.0)], #FP > m0, stable
        #['Cubic', dict(τ = +20, B = -3.2)], #FP > m0, unstable
        #['Cubic', dict(τ = -20, B = +3.2)], #FP > m0, stable
    ]

    for filename, params in cases:
        print(f'Running {filename}')
        params = {**base_params, **params}
        run_and_save(params, filename, logger, m0=m0, c0=c0)

    # Diffusive cases
    base_params = dict(
        mu = 1., α = 3., #Stokes terms, roughly correct units
        Lx=452., Ly=533.,   #Units of microns
        Nx=64, Ny=64,     #Not equal to check transpose
        max_time=100, timestep=0.1, save_every=5,
    )

    D = [1e1, 1e2, 1e3, 1e4, 1e5]
    cases = [
        ['Linear Diffusion', dict(τ = 20)],
        ['Quadratic Diffusion', dict(τ = -20, A = +1)], #Unstable FP < m0
        ['Quadratic Diffusion', dict(τ = +20, A = -1)], #Stable FP < m0
        ['Quadratic Diffusion', dict(τ = -20, A = +0.4)], #Unstable FP > m0
        ['Quadratic Diffusion', dict(τ = +20, A = -0.4)], #Stable FP > m0
        ['Cubic Diffusion', dict(τ = -20, A = 4/3, B=-20/3)]
    ]

    for filename, params in cases:
        params = {**base_params, **params}
        for d in D:
            print(f'Running {filename}, D={d}')
            params['D'] = d
            run_and_save(params, filename, logger, m0=m0, c0=c0)