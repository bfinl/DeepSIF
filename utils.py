import numpy as np
from skimage.filters import threshold_otsu


def ispadding(x):
    # identify the padding in array
    return np.abs(x - 15213) < 1e-6


def get_otsu_regions(out, labels, args_params = None):
    """ Identify DeepSIF source region using otsu_threshould, run on CPU
    :param out: np.arrry; the output of DeepSIF, batch_size * num_time * num_region
    :param labels: np.arrry; group truth source region; batch_size * num_source * max_size; starts from 0
    :param args_params: optional parameters, could be
                        dis_matrix: np.array; distance between regions; num_region (994) * num_region
    :return return_eval: could be
                         all_regions: DeepSIF predicted regions; (batch_size, )
                         all_out:     DeepSIF predicted source activity;  (batch_size, )
    """
    # when there is no spike, the location error is nan

    batch_size = labels.shape[0]
    return_eval = dict()

    return_eval['all_regions'] = np.empty((batch_size,), dtype=object)
    return_eval['all_out'] = np.empty((batch_size,), dtype=object)

    for i in range(batch_size):
        thre_source = np.abs(out[i])
        thre_source = (thre_source - np.min(thre_source)) / np.max(thre_source)
        thresh = threshold_otsu(thre_source, nbins=100)
        select_pixel = out[i] > thresh
        otsu_region = np.where(np.sum(select_pixel, axis=0) > 7)[0]
        return_eval['all_regions'][i] = otsu_region
        return_eval['all_out'][i] = out[i, :, otsu_region]

    # Calculate the eval metrics in Python overall condition for all sources
    if args_params is not None:
        return_eval['precision'] = np.zeros(batch_size)
        return_eval['recall'] = np.zeros(batch_size)
        return_eval['le'] = np.zeros(batch_size)
        for i in range(batch_size):
            lb = labels[i][np.logical_not(ispadding(labels[i]))]
            recon = return_eval['all_regions'][i]
            overlap_region = len(np.intersect1d(lb, recon))
            # number of region based precision and recall
            return_eval['precision'][i] = overlap_region/len(recon)
            return_eval['recall'][i] = overlap_region / len(lb)
            le_each_region = np.min(args_params['dis_matrix'][recon][:, lb], axis = 1)
            return_eval['le'][i] = np.mean(le_each_region)

    return return_eval


def add_white_noise(sig, snr, args_params=None):
    """
    :param sig: np.array; num_electrode * num_time
    :param snr: int; signal to noise level in dB
    :param args_params: optional parameters, could be
                        ratio: np.array; ratio between white Gaussian noise and pre-set realistic noise
                        rndata: np.array; realistic noise data; num_sample * num_electrode * num_time
                        rnpower: np.array; pre-calculated power for rndata; num_sample * num_electrode

    :return: noise_sig: np.array; num_electrode * num_time
    """

    num_elec, num_time = sig.shape
    noise_sig = np.zeros((num_elec, num_time))
    sig_power = np.square(np.linalg.norm(sig, axis=1))/num_time
    if args_params is None:
        # Only add Gaussian noise
        for i in range(num_elec):
            noise_power = 10 ** (-(snr / 10)) * sig_power[i] / 2
            noise_std = np.sqrt(noise_power)
            noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,))
    else:
        # Add realistic and Gaussian noise
        rnpower = args_params['rnpower']/num_time
        rndata = args_params['rndata']
        select_id = np.random.randint(0, rndata.shape[0])
        for i in range(num_elec):
            noise_power = 10 ** (-(snr / 10)) * sig_power[i]
            rpower = args_params['ratio']*noise_power                                 # realistic noise power
            noise_std = np.sqrt(noise_power - rpower)
            noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,)) + np.sqrt(rpower/rnpower[select_id][i])*rndata[select_id][:, i]
    return noise_sig


def fwdJ_to_cortexJ(recon, rm):
    """
    :param recon: np.array; DeepSIF output, (num_time, num_region)
    :param rm: np.array; region mapping for each index, (num_vertices, )
    :return: J: np.array; DeepSIF output for each vertices, (num_time, num_vertices)
    """
    num_time, num_region = recon.shape
    num_vertices = rm.shape[0]
    J = np.zeros((num_time, num_vertices))
    for k in range(num_time):
        for i in range(num_region):
            J[k, rm==i] = recon[k, i]
    return J