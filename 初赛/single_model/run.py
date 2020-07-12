# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os.path as osp
import sys
import torchreid


def main():
    save_dir = 'log/logger_epoch150_labelsmooth121_decay30_and_120'
    log_name = 'log.txt'
    sys.stdout = torchreid.utils.Logger(osp.join(save_dir, log_name))

    loss = 'triplet'
    # 2.Load data manager
    my_dataloader = torchreid.data.datamanager.MyDataManager(root='/project/ywchong/ywchong/CODE/zc/AIC/data/',
                                                             height=384, width=128, 
                                                             # norm_mean=None, norm_std=None, use_gpu=True,
                                                             norm_mean=[0.09721232, 0.18305508, 0.21273703],   # 这里的值是训练集+测试集A的均值
                                                             norm_std=[0.17512791, 0.16554857, 0.22157137],   # 这里的值是训练集+测试集A的方差
                                                             use_gpu=True,
                                                             batch_size_train=64, batch_size_test=128,
                                                             workers=4, num_instances=4,
                                                             loss=loss,
                                                             train_sampler='RandomIdentitySampler',  # if triplet, PK sample is needing
                                                             transforms=['random_flip', 'random_crop_pad', 'random_erase']
                                                             )
    # print(my_dataloader.num_train_pids)  
    # 3. Build model, optimizer and lr_scheduler
    model = torchreid.models.build_model(
        name='se_resnext101_32x4d',  
        num_classes=my_dataloader.num_train_pids,
        loss=loss,
        pretrained=True
    )

    model = model.cuda()
    # 模型并行
    # import torch
    # gpu_ids = [0, 1]
    # torch.cuda.set_device(gpu_ids[0])
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam', 
        lr=0.00035  
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[30, 120]
    )

    # load pre-trained model
    # fpath = "/project/ywchong/ywchong/CODE/zc/AIC/code/log/baseline_senext101/no_rerank/model.pth.tar-120"
    # torchreid.utils.torchtools.resume_from_checkpoint(fpath, model, optimizer)
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
        use_ranked_loss=True, 
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
        eval_freq=100,  
        rerank=False,
        print_freq=100,
        dist_metric='cosine', 
        test_only=False,
        label_smooth_end_epoch=120 
    )


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()