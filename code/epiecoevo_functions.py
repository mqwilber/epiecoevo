from scipy.integrate import odeint
import scipy.optimize as opt
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, Matrix, lambdify

"""

Functions used to analyze the Epi-Eco-Evo model in the manuscript 
"Towards a theory of host recovery dynamics following disease-induced declines"

"""

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def tolerance_model_no_tradeoff_approximation(state_vars, t, p):
    """
    Single-species moment-closure model with tolerance and perfect inheritance,
    approximated to 5 equations.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alpha, v]
        Where N is total population size, I is infected population size,
        Z is pathogen population size in the environment, alpha is the mean 
        tolerance, and v is the second moment in tolerance.
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate

    Returns
    -------
    : right-hand side of ODEs

    """
    
    N, I, Z, alpha, v = state_vars
    S = N - I
    var = v - alpha**2

    dN = p['r']*N - p['delta']*N**2 - p['mu']*N - I*alpha
    dI = p['beta']*Z*S - p['mu']*I - alpha*I
    dZ = p['lam']*I - p['mu_z']*Z
    omega = p['r'] - p['delta']*N
    dalpha = - (I / N)*(v - alpha**2)
    dv = -2*(I / N)*var*(var / alpha + alpha)
    
    return([dN, dI, dZ, dalpha, dv])


def tolerance_model_no_tradeoff(state_vars, t, p):
    """
    Single-species moment-closure model with tolerance and perfect inheritance,
    7 equations.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alphaN, alphaI, vN, vI]
            N is total population size
            I is infected population size,
            Z is pathogen population size in the environment
            alphaN is the mean tolerance in the full population
            alphaI is the mean tolerance in the infected population
            vN is the second moment of tolerance in the full population
            vI is the second moment of tolerance in the infected population
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate

    Returns
    -------
    : right-hand side of ODEs
    """

    N, I, Z, alphaN, alphaI, vN, vI = state_vars
    S = N - I
    alphaI3 = third_moment_gamma(alphaI, vI - alphaI**2)

    dN = p['r']*N - p['delta']*N**2 - p['mu']*N - I*alphaI
    dI = p['beta']*Z*S - p['mu']*I - I*alphaI
    dZ = p['lam']*I - p['mu_z']*Z
    dalphaN = -(I / N)*(vI - alphaN*alphaI)

    if I != 0:
        dalphaI = p['beta']*Z*(N / I)*(alphaN - alphaI) - (vI - alphaI**2)
        dvI = p['beta']*Z*(N / I)*(vN - vI) - (alphaI3 - vI*alphaI)
    else:
        dalphaI = 0
        dvI = 0

    dvN = -(I / N)*(alphaI3 - vN*alphaI)

    return([dN, dI, dZ, dalphaN, dalphaI, dvN, dvI])


def intensity_reduction_resistance_model_with_tradeoff(state_vars, t, p):
    """
    Single-species moment-closure model with tolerance and perfect inheritance,
    7 equations.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alphaN, alphaI, vN, vI]
            N is total population size
            I is infected population size,
            Z is pathogen population size in the environment
            alphaN is the mean tolerance in the full population
            alphaI is the mean tolerance in the infected population
            vN is the second moment of tolerance in the full population
            vI is the second moment of tolerance in the infected population
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate

    Returns
    -------
    : right-hand side of ODEs
    """

    N, I, Z, alphaN, alphaI, vN, vI = state_vars
    S = N - I
    alphaI3 = third_moment_gamma(alphaI, vI - alphaI**2)
    kN = get_k(alphaN, vN)

    dN = p['r']*N*g1(alphaN, kN, p['alpha_m']) - p['delta']*N**2 - p['mu']*N - I*alphaI
    dI = p['beta']*Z*S - p['mu']*I - I*alphaI
    dZ = p['phi']*I*alphaI - p['mu_z']*Z
    dalphaN = p['r']*(g2(alphaN, kN, p['alpha_m']) - alphaN*g1(alphaN, kN, p['alpha_m']))    - (I / N)*(vI - alphaN*alphaI)

    if I != 0:
        dalphaI = p['beta']*Z*(N / I)*(alphaN - alphaI) - (vI - alphaI**2)
        dvI = p['beta']*Z*(N / I)*(vN - vI) - (alphaI3 - vI*alphaI)
    else:
        dalphaI = 0
        dvI = 0

    dvN = p['r']*(g3(alphaN, kN, p['alpha_m']) - vN*g1(alphaN, kN, p['alpha_m'])) - (I / N)*(alphaI3 - vN*alphaI)

    return([dN, dI, dZ, dalphaN, dalphaI, dvN, dvI])


def intensity_reduction_resistance_model_no_tradeoff(state_vars, t, p):
    """
    Single-species moment-closure model with tolerance and perfect inheritance,
    7 equations.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alphaN, alphaI, vN, vI]
            N is total population size
            I is infected population size,
            Z is pathogen population size in the environment
            alphaN is the mean tolerance in the full population
            alphaI is the mean tolerance in the infected population
            vN is the second moment of tolerance in the full population
            vI is the second moment of tolerance in the infected population
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate
            'phi': scaling parameter: ratio of per parasite shdding rate to per parasite mortality rate

    Returns
    -------
    : right-hand side of ODEs
    """

    N, I, Z, alphaN, alphaI, vN, vI = state_vars
    S = N - I
    alphaI3 = third_moment_gamma(alphaI, vI - alphaI**2)

    dN = p['r']*N - p['delta']*N**2 - p['mu']*N - I*alphaI
    dI = p['beta']*Z*S - p['mu']*I - I*alphaI
    dZ = p['phi']*I*alphaI - p['mu_z']*Z
    dalphaN = -(I / N)*(vI - alphaN*alphaI)

    if I > 1e-10:
        dalphaI = p['beta']*Z*(N / I)*(alphaN - alphaI) - (vI - alphaI**2)
        dvI = p['beta']*Z*(N / I)*(vN - vI) - (alphaI3 - vI*alphaI)
    else:
        dalphaI = 0
        dvI = 0


    dvN = -(I / N)*(alphaI3 - vN*alphaI)

    return([dN, dI, dZ, dalphaN, dalphaI, dvN, dvI])


def intensity_reduction_resistance_model_no_tradeoff_depletion(state_vars, t, p):
    """
    Single-species moment-closure model with tolerance and perfect inheritance,
    7 equations, and parasite depletion.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alphaN, alphaI, vN, vI]
            N is total population size
            I is infected population size,
            Z is pathogen population size in the environment
            alphaN is the mean tolerance in the full population
            alphaI is the mean tolerance in the infected population
            vN is the second moment of tolerance in the full population
            vI is the second moment of tolerance in the infected population
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate
            'phi': scaling parameter: ratio of per parasite shdding rate to per parasite mortality rate

    Returns
    -------
    : right-hand side of ODEs
    """

    N, I, Z, alphaN, alphaI, vN, vI = state_vars
    S = N - I
    alphaI3 = third_moment_gamma(alphaI, vI - alphaI**2)

    dN = p['r']*N - p['delta']*N**2 - p['mu']*N - I*alphaI
    dI = p['beta']*Z*S - p['mu']*I - I*alphaI
    dZ = p['phi']*I*alphaI - p['mu_z']*Z - p['beta']*Z*N
    dalphaN = -(I / N)*(vI - alphaN*alphaI)

    if I > 1e-10:
        dalphaI = p['beta']*Z*(N / I)*(alphaN - alphaI) - (vI - alphaI**2)
        dvI = p['beta']*Z*(N / I)*(vN - vI) - (alphaI3 - vI*alphaI)
    else:
        dalphaI = 0
        dvI = 0


    dvN = -(I / N)*(alphaI3 - vN*alphaI)

    return([dN, dI, dZ, dalphaN, dalphaI, dvN, dvI])


def tolerance_model_tradeoff_approximation(state_vars, t, params):
    """
    Single-species moment-closure model with tolerance and perfect inheritance,
    approximated to 5 equations.  Includes piece-wise linear trade-off between
    fecundity and tolerance.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alpha, v]
            N is total population size
            I is infected population size
            Z is pathogen population size in the environment
            alpha is the mean tolerance
            v is the second moment in tolerance.
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate
            'alpha_m': Tolerance at which fecundity is r

    Returns
    -------
    : right-hand side of ODEs

    """
    
    N, I, Z, alpha, v = state_vars
    S = N - I
    k = alpha**2 / (v  - alpha**2)
    var = (v  - alpha**2)

    # Right-hand side
    dN = params['r']*N*g1(alpha, k, params['alpha_m']) - params['delta']*N**2 - params['mu']*N - alpha*I
    dI = params['beta']*S*Z - I*(params['mu'] + alpha)
    dZ = I*params['lam'] - params['mu_z']*Z

    dalpha = params['r']*(g2(alpha, k, params['alpha_m']) - alpha*g1(alpha, k, params['alpha_m'])) - (I / N) * (v - alpha**2)

    dv = (params['r']*(g3(alpha, k, params['alpha_m']) - v*g1(alpha, k, params['alpha_m']))
            - 2*(I / N)*var*((var / alpha) + alpha))

    return([dN, dI, dZ, dalpha, dv])


def resistance_model_tradeoff(state_vars, t, params):
    """
    Single-species moment-closure model with resistance and perfect inheritance.  
    Includes piece-wise linear trade-off between fecundity and resistance.

    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, betaN, betaI vN, vI]
            N is the total population size
            I is the infected population size
            Z is the pathogen population size
            betaN: Mean resistance in the total population
            betaI: Mean resistance in the infected population
            vN: Second moment of resistance in the total population
            vI: Second momemtn of resistance in the infected population 
    t : float
        Time point at which to evaluate the model
    p : dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'alpha': Disease-induced mortality rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate
            'beta_m': Resistance at which fecundity is r

    Returns
    -------
    : list
        Right-hand side of model
    """
    
    p = Struct(**params)
    N, I, Z, betaN, betaI, vN, vI = state_vars

    # Third moment calculations
    thirdN = third_moment_gamma(betaN, (vN - betaN**2))
    thirdI = third_moment_gamma(betaI, (vI - betaI**2))
    kN = get_k(betaN, vN)
    kI = get_k(betaI, vI)
    
    # Right-hand side
    dN = p.r*N*g1(betaN, kN, p.beta_m) - p.delta*N**2 - p.mu*N - p.alpha*I
    
    dI = Z*N*betaN - Z*I*betaI - I*(p.mu + p.alpha)
    
    dbetaN = (p.r*(g2(betaN, kN, p.beta_m) - betaN*g1(betaN, kN, p.beta_m)) + 
             p.alpha*(I / N)*(betaN - betaI))
    
    if I != 0:
        dbetaI = Z*(N / I)*(vN - betaI*betaN) - Z*(vI - betaI**2)
        dvI = Z*(N / I)*(thirdN - vI*betaN) - Z*(thirdI - vI*betaI)
    else:
        dbetaI = 0
        dvI = 0
    
    dvN = p.r*(g3(betaN, kN, p.beta_m) - vN*g1(betaN, kN, p.beta_m)) - (I / N)*p.alpha*(vI - vN) 
    
    dZ = I*p.lam - p.mu_z*Z
    
    return([dN, dI, dZ, dbetaN, dbetaI, dvN, dvI])


def third_moment_gamma(mean, var):
    """
    Third moment of a gamma distribution in terms of mean and var

    Parameters
    ----------
    mean : float
        Mean of distribution
    var : float
        Variance of distribution
    """

    third = (2*(var**2) / mean) + 3*mean*var + mean**3
    return(third)


def moment_closure_model_single_no_evo_equil(state_vars, p):
    """
    Single species model with no evolution. Used to solve for equilibrium.

    Parameters
    ----------
    state_vars : array-like
        N, I, Z
            N: Total population size
            I: Infected population size
            Z: Parasite population size in the environment
    p : dict
        Dictionary of parameters
            'r': Host reproductive rate
            'mu': Host mortality rate
            'delta': Intraspecific competition
            'alpha': Disease-induced mortality rate
            'beta': Transmission rate
            'lam': Parasite reproduction rate
            'mu_z': Parasite death rate

    Returns
    -------
    : array-like
        Right-hand side of ODEs, used to finding equilibria.


    """
    
    N, I, Z = state_vars
    S = N - I

    dN = p['r']*N - p['delta']*N**2 - p['mu']*N - I*p['alpha']
    dI = p['beta']*Z*S - p['mu']*I - p['alpha']*I
    dZ = p['lam']*I - p['mu_z']*Z
    
    return(np.array([dN, dI, dZ]))


def moment_closure_model_single_spp_no_evo(state_vars, t, p):
    """
    Single species model with no evolution. Used to for simulation

    Parameters
    ----------
    state_vars : array-like
        N, I, Z
            N: Total population size
            I: Infected population size
            Z: Parasite population size in the environment
    t : float or array-like
        Time at which to solve equations
    p : dict
        Dictionary of parameters
            'r': Host reproductive rate
            'mu': Host mortality rate
            'delta': Intraspecific competition
            'alpha': Disease-induced mortality rate
            'beta': Transmission rate
            'lam': Parasite reproduction rate
            'mu_z': Parasite death rate

    Returns
    -------
    : array-like
        Right-hand side of ODEs
    """
    
    N, I, Z = state_vars
    S = N - I

    dN = p['r']*N - p['delta']*N**2 - p['mu']*N - I*p['alpha']
    dI = p['beta']*Z*S - p['mu']*I - p['alpha']*I
    dZ = p['lam']*I - p['mu_z']*Z
    
    return([dN, dI, dZ])


def moment_closure_model_tolerance_fecundity_tradeoff_equil(state_vars, params):
    """
    Single species model that includes a tolerance and fecundity tradeoff.
    Set-up to solve for the equilibrium.
    
    Parameters
    ----------
    state_vars : array-like
        [N, I, Z, alpha, v]
            N is total population size
            I is infected population size
            Z is pathogen population size in the environment
            alpha is the mean tolerance
            v is the second moment in tolerance.
    t : float
        Time step
    p: dict
        Dictionary of parameters
            'r': Host fecundity
            'delta': Intraspecific competition
            'mu': Host death rate
            'beta': Tranmission rate
            'lam': Parasite shedding rate
            'mu_z': Parasite death rate
            'alpha_m': Tolerance at which fecundity is r

    Returns
    -------
    : right-hand side of ODE
    """
    
    N, I, Z, alpha, v = state_vars
    S = N - I
    k = alpha**2 / (v  - alpha**2)
    var = v  - alpha**2

    # Right-hand side
    dN = params['r']*N*g1(alpha, k, params['alpha_m']) - params['delta']*N**2 - params['mu']*N - alpha*I
    dI = params['beta']*S*Z - I*(params['mu'] + alpha)
    dZ = I*params['lam'] - params['mu_z']*Z

    dalpha = params['r']*(g2(alpha, k, params['alpha_m']) - alpha*g1(alpha, k, params['alpha_m'])) - (I / N) * (v - alpha**2)

    dv = (params['r']*(g3(alpha, k, params['alpha_m']) - v*g1(alpha, k, params['alpha_m']))
          - 2*(I / N)*var*((var / alpha) + alpha))

    return([dN, dI, dZ, dalpha, dv])




def get_k(mu, v):
    """
    Return k of the gamma distribution from mean (mu) and variance (var)
    """
    
    return(mu**2 / (v  - mu**2))
        

def third_moment(gamma, sigma, mu):
    """
    Third modment of gamma distribution

    Parameters
    ----------
    gamma : float
        skew
    sigma : float
        Standard deviation
    mu : float
        mean
    """
    return(gamma*sigma**3 + 3*mu*sigma**2 + mu**3)


def g1(beta, k, beta_m):
    """
    Piecewise tradeoff function
    """
    
    
    if beta_m != 0:
        val = (beta / beta_m)*stats.gamma.cdf(beta_m, a=k + 1, scale=beta / k) + (1 - stats.gamma.cdf(beta_m, a=k, scale=beta / k))
    else:
        val = 1
        
    return(val)
    
    
def g2(beta, k, beta_m):
    """
    Piecewise tradeoff function
    """
    
    theta = beta / k
    
    if beta_m != 0:
        val = (((k + 1)*k*theta**2) / beta_m)*stats.gamma.cdf(beta_m, a=k + 2, scale=theta) + k*theta*(1 - stats.gamma.cdf(beta_m, a=k + 1, scale=theta))
    else:
        val = k*theta
    return(val)


def g3(beta, k, beta_m):
    """
    Piecewise tradeoff function
    """
    
    theta = beta / k
    
    if beta_m != 0:
        val = (((k + 2)*(k + 1)*k*theta**3) / beta_m)*stats.gamma.cdf(beta_m, a=k + 3, scale=theta) + (k + 1)*k*theta**2*(1 - stats.gamma.cdf(beta_m, a=k + 2, scale=theta))
    else:
        val = (k + 1)*k*theta**2
    return(val)


def max_eig(N, params):
    """
    Intrinsic parasite growth rate at disease free equilibrium
    """

    b = params['alpha'] + params['mu']
    t1 = -((b + params['mu_z']))
    t2 = np.sqrt(4*N*params['beta']*params['lam'] + b**2 - 2*b*params['mu_z'] + params['mu_z']**2)
    full = (1 / 2) * (t1 + t2)
    return(full)


def recovery_metrics(N, time, percent=0.05, baseline=None, change_point=True):
    """
    Compute the four metrics of recovery from a population trajectory
    
    magnitude of decline, time until decline stops, time until full recovery
    
    Parameters
    ----------
    N : array-like
        The population trajectory from the simulation
    time : array-like
        Same length as N corresponding to time
    percent : float
        Percent that N needs to be within the baseline for "full recovery"
    baseline : float
        If None, uses N[0] as baseline recovery value, otherwise use baseline
    change_point : bool
        If True, calculate point of recovery with change in slope. If False,
        calculate with minimum.
    
    Returns
    -------
    : list with recovery metrics
        magnitude of decline, time until decline stops, time until full recovery
    
    """
    
    deltat = time[1] - time[0]

    # What is the index where recovery starts to occur?

    if change_point:
        vals = np.sign(np.diff(N)) == -1
        other_vals = vals[1:]
        all_inflection_points = np.where(np.bitwise_and(vals[:-1], ~other_vals))[0]
        sign_ind = all_inflection_points # Get the final inflection point
    else:
        sign_ind = np.atleast_1d(np.nanargmin(N))

    if baseline is None:
        baseline = N[0]
    
    # Check that there is some decline
    if len(sign_ind) > 0:
        
        # This is the last inflection point (or the first if length is 1)
        ind = sign_ind[-1]

        # Calculate when the population has recovered to within some percent of the original equilibrium
        # Baseline can be whatever value is of interest.
        recov_inds = np.where(1 - (N / baseline) < percent)[0]
        
        if np.any(recov_inds > ind):
            recov_ind_final = np.min(recov_inds[recov_inds > ind])
            time_until_full_recovery = time[recov_ind_final] - time[ind]
        else:
            time_until_full_recovery = np.nan
        
        time_until_decline_stops = time[ind]
        mag_of_decline = 1 - (N[ind] / N[0])

        metrics = [mag_of_decline, time_until_decline_stops, time_until_full_recovery]
    else:
        metrics = [np.nan, np.nan, np.nan]
        
    return(metrics)



### Multispecies models ###


def multispecies_epiecoevo(state_vars, t, p):
    """
    A community model with resistant and tolerant species. 

    In state_vars, all resistant individuals and state variables must be 
    specified first, such as 
    [N, I, betaN, betaI, vN_beta, vI_beta, N, I, betaN, betaI, vN_beta, vI_beta, ..etc.]

    Tolerant individual are specified after resistant individuals such as

    [N, I, betaN, betaI, vN_beta, vI_beta, N, I, alphaN, vN_alpha] where the
    first six state variables are a resistant species and the second four state
    variables are tolerant species.

    The final state variable should be 'Z', the pathogen density in the 
    environment

    The parameters dict p needs needs to specify how many resistant and tolerant
    species are in the community with `num_resistant` and `num_tolerant`, respectively. 

    In addition, 

    Parameters
    ----------
    state_vars : vector
        See description above for ordering
    t : float
        Time
    p : dict
        Parameters for the model. Must include
        'num_resistant': Number of resistance hosts in the community
        'num_tolerant': Number of tolerant hosts in the community
        'beta_m': Vector of beta_m values for fecundity-resistance tradeoffs in
                  resistant hosts 
        'alpha_m': Vector of alpha_m values for fecundity-tolerance tradeoffs
                   in tolerant hosts.
        'r': Vector of fecundity rates for all species (resistant + tolerant)
        'mu': Vector of mortality rates for all species (resistant + tolerant)
        'delta': Matrix of inter-specific competition coefficients
        'alpha': Vector of fixed disease-induced mortality values for resistance species
        'beta': Vector of fixed transmission rates for tolerant species
        'lam': Vector of parasite shedding rates for all species
        'mu_z': Float that of the parasite mortality rate in the environment

    Returns
    -------
    : list
        Right-hand side of the multi-species ODE

    """

    # Resistant species first, tolerant species second
    num_resistant = p['num_resistant']
    num_tolerant = p['num_tolerant']

    # Resistant hosts come first
    # There are six state variables for resistant hosts
    si_resistant = np.arange(0, 6*num_resistant)[::6]

    # Tolerant hosts come second
    # There are four state variables for tolerant
    # Add num_resistant because you need to start *after* these species
    si_tolerant = np.arange(0, 4*num_tolerant)[::4] + 6*num_resistant 

    # Extract N for all hosts
    Nvals = [state_vars[i] for i in np.r_[si_resistant, si_tolerant]]
    Z = state_vars[-1]

    rhs = []

    # Loop through resistant hosts
    spp_count = 0
    for r, start in enumerate(si_resistant):

        N, I, betaN, betaI, vN_beta, vI_beta = state_vars[start:(start + 6)]

        thirdN_beta = third_moment_gamma(betaN, (vN_beta - betaN**2))
        thirdI_beta = third_moment_gamma(betaI, (vI_beta - betaI**2))
        kN_beta = get_k(betaN, vN_beta)
        kI_beta = get_k(betaI, vI_beta)

        dN = p['r'][spp_count]*N*g1(betaN, kN_beta, p['beta_m'][r]) - N*(np.sum(Nvals*p['delta'][spp_count, :])) - p['mu'][spp_count]*N - p['alpha'][r]*I
        
        dI = Z*N*betaN - Z*I*betaI - I*(p['mu'][spp_count] + p['alpha'][r])
        
        dbetaN = (p['r'][spp_count]*(g2(betaN, kN_beta, p['beta_m'][r]) - betaN*g1(betaN, kN_beta, p['beta_m'][r])) + 
                 p['alpha'][r]*(I / N)*(betaN - betaI))

        if I != 0:
            dbetaI = Z*(N / I)*(vN_beta - betaI*betaN) - Z*(vI_beta - betaI**2)
            dvI_beta = Z*(N / I)*(thirdN_beta - vI_beta*betaN) - Z*(thirdI_beta - vI_beta*betaI)
        else:
            dbetaI = 0
            dvI_beta = 0
        
        dvN_beta = p['r'][spp_count]*(g3(betaN, kN_beta, p['beta_m'][r]) - vN_beta*g1(betaN, kN_beta, p['beta_m'][r])) - (I / N)*p['alpha'][r]*(vI_beta - vN_beta) 

        spp_count += 1
        rhs.append([dN, dI, dbetaN, dbetaI, dvN_beta, dvI_beta])

    # Loop through tolerant hosts
    for t, start in enumerate(si_tolerant):

        N, I, alphaN, vN_alpha = state_vars[start:(start + 4)]

        kN_alpha = alphaN**2 / (vN_alpha  - alphaN**2)
        var_alpha = (vN_alpha - alphaN**2)

        dN = p['r'][spp_count]*N*g1(alphaN, kN_alpha, p['alpha_m'][t]) - N*(np.sum(Nvals*p['delta'][spp_count, :])) - p['mu'][spp_count]*N - alphaN*I
        dI = p['beta'][t]*(N - I)*Z - I*(p['mu'][spp_count] + alphaN)

        dalphaN = p['r'][spp_count]*(g2(alphaN, kN_alpha, p['alpha_m'][t]) - alphaN*g1(alphaN, kN_alpha, p['alpha_m'][t])) - (I / N) * (vN_alpha - alphaN**2)

        dvN_alpha = (p['r'][spp_count]*(g3(alphaN, kN_alpha, p['alpha_m'][t]) - vN_alpha*g1(alphaN, kN_alpha, p['alpha_m'][t]))
                - 2*(I / N)*var_alpha*((var_alpha / alphaN) + alphaN))

        spp_count += 1
        rhs.append([dN, dI, dalphaN, dvN_alpha])
        

    Ivals = [state_vars[i + 1] for i in np.r_[si_resistant, si_tolerant]]
    dZ = np.sum(p['lam']*Ivals) - p['mu_z']*Z

    # Format final results
    rhs.append(list(dZ))
    res = list(np.concatenate(rhs))

    return(res)


def disease_free_equil(state_vars, p):
    """
    Find disease-free equilibrium densities for multi-species model
    """
    
    # Extract state variables
    num_spp = len(state_vars)
    N = state_vars
    
    dN = []

    for i in range(num_spp):
        dN_i = p['r'][i]*N[i] - N[i]*np.sum(p['delta'][i,:]*N) - p['mu'][i]*N[i]
        dN.append(dN_i)
        
    return(dN)


def shannon_diversity(Nvals):
    """
    Shannon diversity from abundances

    Parameters
    ----------
    Nvals : array-like
        Vector of host abundances in community

    Returns
    -------
    : float
        Shannon diversity
    """

    p_i = Nvals / np.sum(Nvals)
    H = -np.sum(p_i * np.log(p_i))
    return(H)

def full_resistance_model(state_vars, t, p):
    """
    This is an approximation of the full resistance model without the moment
    closure approach.  In this model, each bin of N and I have a different
    value of beta.

    Parameters
    ----------
    state_vars : array-like
        N vector, I vector, Z where N and I vector the different beta classes
    t : array
        time
    p : dict
        Parameters including
        'bins': The number of discrete beta classes to use
        'beta_vals': The vector of beta values that correspond to each class
        'r_vals': The vector of r values that correspond to each class
        'delta': Intraspecific competition
        'mu': Host death rate
        'lam': Parasite shedding rate
        'mu_z': Parasite death rate
        'alpha': Disease-induced mortality rate
    """

    bins = p['bins']
    Nvals = state_vars[:bins]
    Ivals = state_vars[bins:(2*bins)]
    Z = state_vars[-1]
    beta_vals = p['beta_vals'] # vector of beta values
    r_vals = p['r_vals'] # Vector of R vals based on trade-off

    Neqs = r_vals*Nvals - p['delta']*np.sum(Nvals)*Nvals - p['mu']*Nvals - p['alpha']*Ivals
    Ieqs = beta_vals*Z*(Nvals - Ivals) - Ivals*(p['mu'] + p['alpha'])
    Zeq = np.sum(Ivals*p['lam']) - p['mu_z']*Z

    rhs = list(np.concatenate([Neqs, Ieqs, [Zeq]]))
    return(rhs)


def full_tolerance_model(state_vars, t, p):
    """
    An approximation of the full tolerance model without the moment
    closure approach.  In this model, each bin of N and I have a different
    value of alpha.

    Parameters
    ----------
    state_vars : array-like
        N vector, I vector, Z where N and I vector the different beta classes
        N vector and I vector should have the same length and are discrete
        bins corresponding to a level of alpha
    t : array
        time
    p : dict
        Parameters including
        'bins': The number of discrete beta classes to use
        'alpha_vals': The vector of alpha values that correspond to each class in N and I
        'r_vals': The vector of r values that correspond to each class in N and I
        'delta': Intraspecific competition
        'mu': Host death rate
        'beta': Transmission rate
        'lam': Parasite shedding rate
        'mu_z': Parasite death rate

    Returns
    -------
    : right-hand side of ODE
    """

    bins = p['bins']
    Nvals = state_vars[:bins]
    Ivals = state_vars[bins:(2*bins)]
    Z = state_vars[-1]
    alpha_vals = p['alpha_vals'] # vector of alpha values
    r_vals = p['r_vals'] # Vector of r vals based on trade-off

    Neqs = r_vals*Nvals - p['delta']*np.sum(Nvals)*Nvals - p['mu']*Nvals - alpha_vals*Ivals
    Ieqs = p['beta']*Z*(Nvals - Ivals) - Ivals*(p['mu'] + alpha_vals)
    Zeq = np.sum(Ivals*p['lam']) - p['mu_z']*Z

    rhs = list(np.concatenate([Neqs, Ieqs, [Zeq]]))
    return(rhs)


def full_ir_resistance_model(state_vars, t, p):
    """
    An approximation of the full intensity-reduction resistance model without the moment
    closure approach.  In this model, each bin of N and I have a different
    value of alpha.

    Parameters
    ----------
    state_vars : array-like
        N vector, I vector, Z where N and I vector the different beta classes
        N vector and I vector should have the same length and are discrete
        bins corresponding to a level of alpha
    t : array
        time
    p : dict
        Parameters including
        'bins': The number of discrete beta classes to use
        'alpha_vals': The vector of alpha values that correspond to each class in N and I
        'r_vals': The vector of r values that correspond to each class in N and I
        'delta': Intraspecific competition
        'mu': Host death rate
        'beta': Transmission rate
        'lam_vals': Parasite shedding rate based on level of alpha
        'mu_z': Parasite death rate

    Returns
    -------
    : right-hand side of ODE
    """

    bins = p['bins']
    Nvals = state_vars[:bins]
    Ivals = state_vars[bins:(2*bins)]
    Z = state_vars[-1]
    alpha_vals = p['alpha_vals'] # vector of alpha values
    r_vals = p['r_vals'] # Vector of r vals based on trade-off

    Neqs = r_vals*Nvals - p['delta']*np.sum(Nvals)*Nvals - p['mu']*Nvals - alpha_vals*Ivals
    Ieqs = p['beta']*Z*(Nvals - Ivals) - Ivals*(p['mu'] + alpha_vals)
    Zeq = np.sum(Ivals*p['lam_vals']) - p['mu_z']*Z

    rhs = list(np.concatenate([Neqs, Ieqs, [Zeq]]))
    return(rhs)


def r_fxn(beta, beta_m, r):
    """ 

    Resistance or tolerance - fecundity trade-off 

    Parameters
    ----------
    beta : float or array
        Parameter value at which to calculate trade-off
    beta_m : float
        The value of beta at which fecundity starts to decrease
    r : float
        The maximum reproductive rate when beta > beta_m

    """
    beta = np.atleast_1d(beta)
    if beta_m > 0:

        newr = np.empty(len(beta))
        ind = beta < beta_m
        newr[ind] = (r / beta_m) * beta[ind]
        newr[~ind] = r

    else:
        newr = np.repeat(r, len(beta))

    return(newr)

def r_fxn_plaw(beta, beta_m, r, slope):
    """
    Resistance or tolerance - fecundity trade-off with piece-wise powerlaw
    tradeoff

    Parameters
    ----------
    beta : float or array
        Parameter value at which to calculate trade-off
    beta_m : float
        The value of beta at which fecundity starts to decrease
    r : float
        The maximum reproductive rate when beta > beta_m
    slope : float
        Slope of the power law function

    """

    beta = np.atleast_1d(beta)
    a = r / (beta_m**slope)
    if beta_m > 0:

        newr = np.empty(len(beta))
        ind = beta < beta_m
        newr[ind] = a*beta[ind]**slope
        newr[~ind] = r

    else:
        newr = np.repeat(r, len(beta))

    return(newr)




