import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import collections

import torch
from torch.utils.data import DataLoader

import network
import loaders
from utils import get_otsu_regions


def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device running the code')
    parser.add_argument('--dat', default='SpikeEEGBuildEval', type=str, help='data loader')
    parser.add_argument('--test', default='test_sample_source2.mat', type=str, help='test dataset name')
    parser.add_argument('--model_id', type=int, default=75, help='model id')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')

    parser.add_argument('--snr_rsn_ratio', default=0, type=float, help='ratio between real noise and gaussian noise')
    parser.add_argument('--lfreq', default=-1, type=int, help='filter EEG data to perform narrow-band analysis')
    parser.add_argument('--hfreq', default=-1, type=int, help='filter EEG data to perform narrow-band analysis')
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = (False) and torch.cuda.is_available()  # Only use GPU during training
    device = torch.device(args.device if use_cuda else "cpu")

    data_root = 'source/Simulation/'
    dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']

    result_root = 'model_result/{}_the_model'.format(args.model_id)
    if not os.path.exists(result_root):
        print("ERROR: No model {}".format(args.model_id))
        return
    fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd']

    # ================================== LOAD DATA ===================================================================================================
    test_data = loaders.__dict__[args.dat](data_root + args.test, fwd=fwd,
                                                args_params={'snr_rsn_ratio': args.snr_rsn_ratio,
                                                             'lfreq': args.lfreq, 'hfreq': args.hfreq})
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # =============================== LOAD MODEL =====================================================================================================
    if args.resume:
        fn = fn = os.path.join(result_root, 'epoch_' + args.resume)
    else:
        fn = os.path.join(result_root, 'model_best.pth.tar')
    print("=> Load checkpoint", fn)
    if os.path.isfile(fn):
        print("=> Found checkpoint '{}'".format(fn))
        checkpoint = torch.load(fn, map_location=torch.device('cpu'))
        best_result = checkpoint['best_result']
        net = network.__dict__[checkpoint['arch']](*checkpoint['attribute_list']).to(device)  # redefine the weights architecture
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> Loaded checkpoint {}, current results: {}".format(fn, best_result))

        # Define logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(result_root + '/outputs_{}.log'.format(checkpoint['arch']))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("=================== Evaluation mode: {} ====================================".format(datetime.datetime.now()))
        logger.info("Testing data is {}".format(args.test))
        # Save every parameters in args
        for v in args.__dict__:
            if v not in ['workers', 'train', 'test']:
                logger.info('{} is {}'.format(v, args.__dict__[v]))
    else:
        print("ERROR: no checkpoint found")
        return

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # =============================== EVALUATION =====================================================================================================
    net.eval()

    eval_dict = collections.defaultdict(list)
    eval_dict['all_out'] = []                                       # DeepSIF output
    eval_dict['all_nmm'] = []                                       # Ground truth source activity
    eval_dict['all_regions'] = []                                   # DeepSIF identified source regions
    eval_dict['all_loss'] = 0                                       # MSE Loss
    criterion = torch.nn.MSELoss(reduction='sum')

    with torch.no_grad():

        for batch_idx, sample_batch in enumerate(test_loader):

            if batch_idx > 0:
                break

            data = sample_batch['data'].to(device, torch.float)
            nmm = sample_batch['nmm'].numpy()
            label = sample_batch['label'].numpy()
            model_output = net(data)
            out = model_output['last']
            # calculate loss function
            # nmm_torch = sample_batch['nmm'].to(device, torch.float)
            # eval_dict['all_loss'] = eval_dict['all_loss'] + criterion(out, nmm_torch).data.numpy()
            # ----- SAVE EVERYTHING TO EXAMINE LATER (not suitable for large test dataset) -------
            # eval_dict['all_out'].append(out.cpu().numpy())
            # eval_dict['all_eeg'].append(data.cpu().numpy())

            # ----- ONLY SAVE IDENTIFIED REGION --------------------------------------------------
            eval_results = get_otsu_regions(out.cpu().numpy(), label)
            # calculate metrics as a sanity check
            # eval_results = get_otsu_regions(out.cpu().numpy(), label, args_params = {'dis_matrix': dis_matrix})
            # eval_dict['precision'].extend(eval_results['precision'])
            # eval_dict['recall'].extend(eval_results['recall'])
            # eval_dict['le'].extend(eval_results['le'])

            eval_dict['all_regions'].extend(eval_results['all_regions'])
            eval_dict['all_out'].extend(eval_results['all_out'])
            # ------------------------------------------------------------------------------------
            for kk in range(out.size(0)):
                eval_dict['all_nmm'].append(nmm[kk, :, label[kk, :, 0]])  # Only save activity in the center region
                # lb = label[kk, :, :]                                    # Save activities in all source regions
                # eval_dict['all_nmm'].append(nmm[kk, :, lb[np.logical_not(ispadding(lb))]])

    savemat(fn + '_preds_{}{}.mat'.format(args.test[:-4], args.info), eval_dict)


if __name__ == '__main__':
    main()

