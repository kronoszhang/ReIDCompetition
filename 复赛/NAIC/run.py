# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os.path as osp
import sys
import torchreid
import torch



def main():

    save_dir = 'log/exp_demo'   # exp8->64*4change160  exp9->exp8+cj
    log_name = 'log.txt'
    sys.stdout = torchreid.utils.Logger(osp.join(save_dir, log_name))

    loss = 'triplet'
    # 2.Load data manager
    my_dataloader = torchreid.data.datamanager.MyDataManager(root='data/',
                                                             height=384, width=128, 
                                                             norm_mean=[0.17493263, 0.21447995, 0.24703274], norm_std=[0.22062957, 0.20586668, 0.26357092], 
                                                             use_gpu=True,  
                                                             batch_size_train=160, batch_size_test=128*4*2, #64, 128
                                                             workers=16, num_instances=4,
                                                             loss=loss,
                                                             train_sampler='RandomIdentitySampler',  # if triplet, PK sample is needing
                                                             transforms=['random_flip', 'random_crop', 'random_erase', 'color_jitter'],  # try other transfer
                                                             )
    # print(my_dataloader.num_train_pids)   
    # 3. Build model, optimizer and lr_scheduler
    model = torchreid.models.build_model(
        name='se_resnext101_32x4d',   # se_resnext101_32x4d
        num_classes=my_dataloader.num_train_pids,  # 4949 for trainset fu sai, total 9968ID, but we final use 4949 only, thus if load pretrained model, use this
        loss=loss,
        pretrained=True
    )

    #model = model.cuda()
    
    #device_ids = [0, 1]
    #from torch import nn
    #model = model.cuda(device_ids[0])
    #model = nn.DataParallel(model, device_ids=device_ids)
    
    #import torch
    #model = nn.DataParallel(model).to(torch.device('cuda'))
    #model = torch.nn.DataParallel(model.cuda())

    import torch
    gpu_ids = [0, 1]
    torch.cuda.set_device(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam', 
        lr=3.5e-4  
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[30, 120]  
    )

    # load pre-trained model
    fpath = "log/exp7//model.pth.tar-60" 
    torchreid.utils.torchtools.resume_from_checkpoint(fpath, model, optimizer)
    # torchreid.utils.torchtools.load_pretrained_weights(model, fpath)
    
    # 4.Build engine
    engine = torchreid.engine.ImageTripletEngine(
        my_dataloader,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_center=True,
        use_oim=False,
        num_classes=my_dataloader.num_train_pids,
        margin=0.7,
        weight_o=1.5,
        center_lr=0.5, 
        weight_c=0.0005,
        oim_dims=2048,
        center_dims=2048,
        use_ranked_loss=False, 
        weight_r=2,
        weight_t=1,
        use_focal=False, 
        weight_f=10
    )
    # 5.Run training and test
    print(save_dir)
    engine.run(
        save_dir=save_dir,
        normalize_feature=True,
        max_epoch=150,  
        eval_freq=30,  
        rerank=False,
        print_freq=400,
        dist_metric='cosine', # euclidean
        test_only=True,
        label_smooth_end_epoch=121 
    )


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()