from scipy.io import savemat
from tvb.simulator.lab import *
import time
import numpy
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from tvb.simulator import (simulator, models, coupling, integrators, monitors, noise)
# from tvb.simulator.models import jansen_rit_wendling
import jansen_rit_wendling

# 01062020: generate data for JR model with different kind of parameters
# 03232020: Adapt to tvb version 2 for _single_nmm
# Notes: unit in mV, and ms, simulation length is in ms


def run_and_save_data(sim, fpath, folder_name, fname, save=True):
    # run it
    start_time = time.time()
    out = sim.run()
    out = sim.run()
    (t, data), = out
    # save data
    if not os.path.isdir(fpath + folder_name):
        os.mkdir(fpath + folder_name)
    if not os.path.isdir(fpath + 'figures/' + folder_name):
        os.mkdir(fpath + 'figures/' + folder_name)
    if save:
        savemat(fpath + fname + '.mat',{'time':t,'data':data})
    
    trange = range(4000, t.shape[0])  # t.shape[0]
    plt.figure()
    for k in range(data.shape[2]):
        plt.plot(t[trange], data[trange, 1, k, 0] - data[trange, 2, k, 0] - data[trange, 3, k, 0])
        if save:
            plt.savefig(fpath + 'figures/{}.jpg'.format(fname))
        #plt.close()
    end_time = time.time()
    print('Total time used for', fname, ':', end_time - start_time)


# ----------- explore the parameter space for one JR NMM model -----------------------------------
def _single_nmm(p_max, p_min, A, B, C, G, save=True):
    # make sure the input is float
    # notations are following the Wendling 2000 paper
    fpath = 'D:/Database/MySim/JR_find_parameters/'
    folder_name = 'ictal_for_testing'

    p_m = (p_max + p_min)/2
    jrm = jansen_rit_wendling.JansenRit_Wendling(mu=np.array([p_m/1000]), p_max=np.array([p_max/1000]), p_min=np.array([p_min/1000]),
                           v0=np.array(6., dtype=float), A=np.array([A]), B=np.array([B]), G = np.array([G]), J=np.array([C]))
    # Note: v0 is also an very important value
    # I think I should scale A accordingly, where I keeped using value A in my first version simulation
    phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max - jrm.p_min)* 0.5) ** 2 / 2.
    sigma = numpy.zeros(10)
    sigma[6] = phi_n_scaling
    con = connectivity.Connectivity()
    con.weights = np.ones((1,1))
    con.tract_lengths = np.zeros((1, 1))
    con.region_labels = np.array(['None'])
    con.centres = np.array([1., 1., 1.,])
    con.configure()   

    randseed = np.random.randint(100)
    randomStream = numpy.random.mtrand.RandomState(randseed)
    hiss = noise.Additive(random_stream=randomStream, nsig=sigma)  #
    integ = integrators.HeunStochastic(dt=1, noise=hiss)


    sim = simulator.Simulator(
        model=jrm,
        connectivity=con,  # connectivity.Connectivity.from_file(),
        coupling=coupling.Linear(a=np.array([0])),
        integrator=integ,  # the noise added is Gaussian noise
        # integrator=integrators.HeunStochastic(dt=2 ** -2, noise=noise.Additive(nsig=sigma)),  # the noise added is Gaussian noise
        # integrator=integrators.RungeKutta4thOrderDeterministic(dt=2 ** -1),
        monitors=(monitors.Raw(),),
        simulation_length= 1e3#5e4
    ).configure()
    # run it
    fname = folder_name + '/p_m_{:02.0f}_p_v_{:02.0f}_a_{:02.2f}_b_{:02.0f}_c_{:02.0f}_g_{:02.0f}'.format(p_m, p_max-p_min, A, B, C, G)
    run_and_save_data(sim, fpath, folder_name, fname, save=save)


def test_single_nmm():
    # test function _single_nmm
    p_min = 30
    p_max = 150
    C = 135.0
    A = 5.5
    B = 11.0#48.557
    G = 10.0
    _single_nmm(p_max, p_min, A, B, C, G, save=False)


def run_single_nmm():
    # run function _single_nmm in parallel
    B_list = [11,13,15,17,19,21,23,25,27,29,30]
    G_list = [5,10,20,25]
    # b_all = np.arange(2,60,2); % parameter range for B
    # g_all = np.arange(2,60,2); % parameter range for G
    for G in G_list:
        # we cannot put all the processes all together
        processes = []
        for B in B_list:
                processes.append(mp.Process(target=_single_nmm, args=(150.0, 30.0, 4.5, B, 135.0, G, True)))

        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()

    # BG_SETS = {3.5:[[15,30]],4:[[10,20],[15,20],[20,20]],4.5:[[13,5],[13,15],[13,25],[15,25],[17,25],[19,25],[25,25],[30,25]],5.5:[[11,20],[15,20],[20,20],[30,20],[40,20]]}
    # for A in BG_SETS:
    #     # we cannot put all the processes all together
    #     processes = []
    #     for B, G in BG_SETS[A]:
    #             processes.append(mp.Process(target=_single_nmm, args=(150.0, 30.0, A, B, 135.0, G, True)))

    #     for p in processes:
    #         p.start()
    #     # Exit the completed processes
    #     for p in processes:
    #         p.join()

 
if __name__ == '__main__':
    startt = time.time()
    test_single_nmm()
    print('Total time used', time.time()-startt)