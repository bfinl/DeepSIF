import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import glob

import torch
import network


def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--device', default='cpu', type=str, help='device running the code')
    parser.add_argument('--model_id', type=int, default=64, help='model id')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = (False) and torch.cuda.is_available()  # Only use GPU during training
    device = torch.device(args.device if use_cuda else "cpu")
    result_root = 'model_result/{}_the_model'.format(args.model_id)
    if not os.path.exists(result_root):
        print("ERROR: No model {}".format(args.model_id))
        return

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
    else:
        print("ERROR: no checkpoint found")
        return

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # =============================== EVALUATION =====================================================================================================
    net.eval()
    subject_list = ['VEP']
    for pii in subject_list:
        folder_name = 'source/{}'.format(pii)
        start_time = time.time()
        flist = glob.glob(folder_name + '/data*.mat')
        if len(flist) == 0:
            print('WARNING: NO FILE IN FOLDER {}.'.format(folder_name))
            continue
        flist = sorted(flist, key=lambda name: int(os.path.basename(name)[4:-4]))  # sort file based on nature number
        test_data = []
        for i in flist:
            data = loadmat(i)['data']
            # data = data - np.mean(data, 0, keepdims=True)
            # data = data - np.mean(data, 1, keepdims=True)
            data = data / np.max(np.abs(data[:]))
            test_data.append(data)

        data = torch.from_numpy(np.array(test_data)).to(device, torch.float)
        out = net(data)['last']
        # calculate the loss
        all_out = out.detach().cpu().numpy()
        # visualize the result in Matlab
        savemat(folder_name + '/rnn_test_{}_{}.mat'.format(args.model_id, fn[-8:]), {'all_out': all_out})
        print('Save output as:', folder_name + '/rnn_test_{}_{}.mat'.format(args.model_id, fn[-8:]))
    print('Total run time:', time.time() - start_time)

if __name__ == '__main__':
    main()

