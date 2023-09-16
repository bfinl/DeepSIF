import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

import network
import loaders

# Update: add loading trained generic model as an option. The file location can be passed as the gmodel parameter

def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--save', type=int, default=True, help='save each epoch or not')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device running the code')
    parser.add_argument('--arch', default='TemporalInverseNet', type=str, help='network achitecture class')
    parser.add_argument('--dat', default='SpikeEEGBuild', type=str, help='data loader')
    parser.add_argument('--train', default='test_sample_source2.mat', type=str, help='train dataset name or directory')
    parser.add_argument('--test', default='test_sample_source2.mat', type=str, help='test dataset name or directory')
    parser.add_argument('--model_id', default=75, type=int, help='model id')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--epoch', default=20, type=int, help='total number of epoch')
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use')
    parser.add_argument('--rnn_layer', default=3, type=int, help='number of rnn layer')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    parser.add_argument('--gmodel', default='', type=str, help='generic model saved location')
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    data_root = 'source/Simulation/'
    result_root = 'model_result/{}_the_model'.format(args.model_id)
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd']

    # Define logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(result_root + '/outputs_{}.log'.format(args.arch))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("============================= {} ====================================".format(datetime.datetime.now()))
    logger.info("Training data is {}, and testing data is {}".format(args.train, args.test))
    # Save every parameters in args
    for v in args.__dict__:
        if v not in ['workers', 'train', 'test']:
            logger.info('{} is {}'.format(v, args.__dict__[v]))

    # ================================== LOAD DATA ===================================================================================================
    train_data = loaders.__dict__[args.dat](data_root + args.train, fwd=fwd,
                                                args_params={'dataset_len': 4})
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    test_data = loaders.__dict__[args.dat](data_root + args.test, fwd=fwd, args_params={'dataset_len': 4})
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # ================================== CREATE MODEL ================================================================================================

    net = network.__dict__[args.arch](num_sensor=75, num_source=994, rnn_layer=args.rnn_layer,
                                      spatial_model=network.MLPSpatialFilter,
                                      temporal_model=network.TemporalFilter,
                                      spatial_output='value_activation', temporal_output='rnn', spatial_activation='ELU', temporal_activation='ELU',
                                      temporal_input_size=500).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, threshold=0.001)
    criterion = torch.nn.MSELoss(reduction='sum')

    args.start_epoch = 0
    best_result = np.Inf
    train_loss = []
    test_loss = []

    # =============================== RESUME =========================================================================================================
    if args.gmodel:
        checkpoint = torch.load(args.gmodel, map_location=torch.device('cpu'))   
        net = Network.__dict__[checkpoint['arch']](*checkpoint['attribute_list']).to(device)  #[:-1]
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> Loaded previous model:{}".format(args.gmodel))

    if args.resume:
        print("=> Load checkpoint", args.resume, "from", result_root)
        fn = os.path.join(result_root, 'epoch_' + args.resume)
        if os.path.isfile(fn):
            print("=> Found checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(fn, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            # recreate net and optimizer based on the saved model
            net = network.__dict__[checkpoint['arch']](*checkpoint['attribute_list']).to(device)  # redefine the weights architecture
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = param_group['lr'] / checkpoint['lr'] * args.lr
            print("=> Loaded checkpoint epoch {}, current results: {}".format(args.resume, best_result))
            tte = loadmat(result_root + '/train_test_error.mat')
            train_loss.extend(tte['train_loss'][0][:int(args.resume) + 1])
            test_loss.extend(tte['test_loss'][0][:int(args.resume) + 1])
        else:
            print("WARNING: no checkpoint found at '{}', use random weights".format(args.resume))

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # =============================== TRAINING =======================================================================================================
    for epoch in range(args.start_epoch + 1, args.epoch):

        # train for one epoch
        train_lss_all = train(train_loader, net, criterion, optimizer, {'device': device, 'logger': logger})
        # evaluate on validation set
        test_lss_all = validate(test_loader, net, criterion, {'device': device})
        # lr_scheduler.step(test_le)
        train_loss.extend([np.sum(np.array(train_lss_all)) / len(train_data)])
        test_loss.extend([np.sum(np.array(test_lss_all))/len(test_data)])

        print_s = 'Epoch {}: Time:{:6.2f}, '.format(epoch, time.time() - start_time) + \
                  'Train Loss:{:06.5f}'.format(train_loss[-1]) + ', Test Loss:{:06.5f}'.format(test_loss[-1])
        logger.info(print_s)
        print(print_s)
        is_best = test_loss[-1] < best_result
        best_result = min(test_loss[-1], best_result)
        if is_best:
            torch.save({
                'epoch': epoch, 'arch': args.arch, 'state_dict': net.state_dict(), 'best_result': best_result, 'lr': args.lr, 'info': args.info,
                'train': args.train, 'test': args.test, 'attribute_list': net.attribute_list, 'optimizer': optimizer.state_dict()},
                result_root + '/model_best.pth.tar')
        if args.save:
            # save checkpoint
            torch.save({
                'epoch': epoch, 'arch': args.arch, 'state_dict': net.state_dict(), 'best_result': best_result, 'lr': args.lr, 'info': args.info,
                'train': args.train, 'test': args.test, 'attribute_list': net.attribute_list, 'optimizer': optimizer.state_dict()},
                result_root + '/epoch_{}'.format(epoch))
            savemat(result_root + '/train_test_error.mat', {'train_loss': train_loss, 'test_loss': test_loss})
            savemat(result_root + '/train_test_loss_epoch{}.mat'.format(epoch), {'train_loss': train_lss_all, 'test_loss': test_lss_all})
    # END MAIN_TRAIN


# START TRAIN FUNC
def train(train_loader, model, criterion, optimizer, args_params):
    # args_params: potential parameter inputs, could be "device","logger"

    device = args_params['device']
    logger = args_params['logger']
    # switch to train mode
    model.train()
    train_loss = []
    start_time = time.time()
    for batch_idx, sample_batch in enumerate(train_loader):
        # load data
        data = sample_batch['data'].to(device)
        nmm = sample_batch['nmm'].to(device)

        # training process
        optimizer.zero_grad()
        model_output = model(data)
        out = model_output['last']
        loss = criterion(out, nmm)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.data.view(1))
        if (batch_idx + 1) % 500 == 0:
            print_s = "batch_idx_{}_time_{}_train_loss_{}".format(batch_idx, time.time() - start_time, train_loss[-1])
            logger.info(print_s)
    train_loss = torch.cat(train_loss).cpu().numpy()
    return train_loss
# END TRAIN


# START VALIDATE FUNC
def validate(val_loader, model, criterion, args_params):
    # switch to evaluate mode
    device = args_params['device']
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(val_loader):
            data = sample_batch['data'].to(device)
            nmm = sample_batch['nmm'].to(device)
            model_output = model(data)
            out = model_output['last']
            loss = criterion(out, nmm)
            val_loss.append(loss.data.view(1))
    val_loss = torch.cat(val_loss).cpu().numpy()
    return val_loss
# END VALIDATE


if __name__ == '__main__':
    main()

