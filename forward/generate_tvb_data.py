from scipy.io import savemat
from tvb.simulator.lab import *
import time
import numpy as np
import multiprocessing as mp
import os
import argparse


def main(region_id):
    """ TVB Simulation to generate raw source space dynamics, unit in mV, and ms
    :param region_id: int; source region id, with parameters generating interictal spike activity
    """
    if not os.path.isdir('../source/raw_nmm/a{}/'.format(region_id)):
        os.mkdir('../source/raw_nmm/a{}/'.format(region_id))
    start_time = time.time()
    print('------ Generate data of region_id {} ----------'.format(region_id))
    conn = connectivity.Connectivity.from_file(source_file=os.getcwd()+'/../anatomy/connectivity_76.zip') # connectivity provided by TVB
    conn.configure()

    # define A value
    num_region = conn.number_of_regions
    a_range = [3.5]
    A = np.ones((num_region, len(a_range))) * 3.25                                  # the normal A value is 3.25
    A[region_id, :] = a_range

    # define mean and std
    mean_and_std = np.array([[0.087, 0.08, 0.083], [1, 1.7, 1.5]])
    for iter_a in range(A.shape[1]):
        use_A = A[:, iter_a]
        for iter_m in range(mean_and_std.shape[1]):

            jrm = models.JansenRit(A=use_A, mu=np.array(mean_and_std[0][iter_m]),
                                   v0=np.array([6.]), p_max=np.array([0.15]), p_min=np.array([0.03]))
            phi_n_scaling = (jrm.a * 3.25 * (jrm.p_max - jrm.p_min) * 0.5 * mean_and_std[1][iter_m]) ** 2 / 2.
            sigma = np.zeros(6)
            sigma[4] = phi_n_scaling

            # set the random seed for the random intergrator
            # randomStream = np.random.mtrand.RandomState(0)
            # noise_class = noise.Additive(random_stream=randomStream, nsig=sigma)
            # integ = integrators.HeunStochastic(dt=2 ** -1, noise=noise_class)

            sim = simulator.Simulator(
                model=jrm,
                connectivity=conn,
                coupling=coupling.SigmoidalJansenRit(a=np.array([1.0])),
                integrator=integrators.HeunStochastic(dt=2 ** -1, noise=noise.Additive(nsig=sigma)),
                monitors=(monitors.Raw(),)
            ).configure()

            # run 200s of simulation, cut it into 20 pieces, 10s each. (Avoid saving large files)
            for iii in range(20):
                siml = 1e4
                out = sim.run(simulation_length=siml)
                (t, data), = out
                data = (data[:, 1, :, :] - data[:, 2, :, :]).squeeze().astype(np.float32)

                # # in the fsaverage5 mapping, there is no vertices corresponding to region 7,325,921, 949, so change label 994-998 to those id
                # data[:, 7] = data[:, 994]
                # data[:, 325] = data[:, 997]
                # data[:, 921] = data[:, 996]
                # data[:, 949] = data[:, 995]
                # data = data[:, :994]

                savemat('../source/raw_nmm/a{}/mean_iter_{}_a_iter_{}_{}.mat'.format(region_id, iter_m, region_id, iii),
                        {'time': t, 'data': data, 'A': use_A})
    print('Time for', region_id, time.time() - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TVB Simulation')
    parser.add_argument('--a_start', type=int, default=0, metavar='t/f', help='start region id')
    parser.add_argument('--a_end', type=int, default=1, metavar='t/f', help='end region id')
    args = parser.parse_args()
    os.environ["MKL_NUM_THREADS"] = "1"
    start_time = time.time()
    # RUN THE CODE IN PARALLEL
    # processes = [mp.Process(target=main, args=(x,)) for x in range(args.a_start, args.a_end)]
    # for p in processes:
    #     p.start()
    # # Exit the completed processes
    # for p in processes:
    #     p.join()
    # NO PARALLEL
    for x in range(args.a_start, args.a_end):
        main(x)
    print('Total_time', time.time() - start_time)

