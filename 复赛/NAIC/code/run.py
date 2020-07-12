# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import sys
import time
import numpy as np
import torchreid
import torch


def hdbscan(feat, min_samples=10):
    import hdbscan
    db = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    labels_ = db.fit_predict(feat)
    return labels_

def main():
    # load pretrained model
    save_dir = 'log/exp_cluster'
    log_name = 'log.txt'
    sys.stdout = torchreid.utils.Logger(osp.join(save_dir, log_name))
    loss = 'triplet'
    my_dataloader = torchreid.data.datamanager.MyDataManager(root='/home/reid/ChronousZ/NAIC/data/',
                                                             height=384, width=128,
                                                             norm_mean=[0.17493263, 0.21447995, 0.24703274],
                                                             norm_std=[0.22062957, 0.20586668, 0.26357092],
                                                             use_gpu=True,
                                                             batch_size_train=64 * 4, batch_size_test=128 * 16,
                                                             # 64, 128
                                                             workers=8, num_instances=4,
                                                             loss=loss,
                                                             train_sampler='RandomIdentitySampler',
                                                             # if triplet, PK sample is needing
                                                             transforms=['random_flip', 'random_crop', 'random_erase', 'color_jitter'],  # try other transfer
                                                             )
    model = torchreid.models.build_model(
        name='se_resnext101_32x4d',
        num_classes=my_dataloader.num_train_pids,
        loss=loss,
        pretrained=True
    )

    # model = model.cuda()
    import torch
    gpu_ids = [0, 1, 2, 3]
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
    fpath = "/home/reid/ChronousZ/NAIC/code/log/exp//model.pth.tar-60"
    # torchreid.utils.torchtools.resume_from_checkpoint(fpath, model, optimizer)
    torchreid.utils.torchtools.load_pretrained_weights(model, fpath)

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


    print(save_dir)
    # engine.run(
    #     save_dir=save_dir,
    #     normalize_feature=True,
    #     max_epoch=150,
    #     eval_freq=30,
    #     rerank=False,
    #     print_freq=200,
    #     dist_metric='cosine',  # euclidean
    #     test_only=True,
    #     label_smooth_end_epoch=121
    # )
    K = 0

    while K < 5:
        start = time.time()
        trainloader, valloader, testloader, testdataset = my_dataloader.trainloader, my_dataloader.valloader, my_dataloader.testloader, my_dataloader.testdataset
        # print(len(trainloader), len(valloader), len(testloader))
        queryloader = testloader['query']
        galleryloader = testloader['gallery']
        qf, gf = engine._evaluate(
            60,
            queryloader=queryloader,
            galleryloader=galleryloader,
            testdataset=testdataset,
            dist_metric='cosine',
            normalize_feature=True,
            save_dir=save_dir,
            return_json=True,
            return_feature=True
        )
        # print(qf.shape, gf.shape)
        test_f = torch.cat((qf, gf), 0)
        # print(test_f.shape)
        print("Extracting feature finished, {:.2f} s time used...".format(time.time()-start))
        # cluster
        start = time.time()
        ofn = osp.join(save_dir, "cluster_result.txt")
        pred = hdbscan(test_f, min_samples=4)
        # post process
        valid = np.where(pred != -1)
        _, unique_idx = np.unique(pred[valid], return_index=True)
        pred_unique = pred[valid][np.sort(unique_idx)]
        pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
        pred_mapping[-1] = -1
        pred = np.array([pred_mapping[p] for p in pred])
        print("Discard ratio: {:.4g}".format(1 - len(valid[0]) / float(len(pred))))
        # save
        with open(ofn, 'w') as f:
            f.writelines(["{}\n".format(l) for l in pred])
        print("Save as: {}".format(ofn))
        num_class_valid = len(np.unique(pred[np.where(pred != -1)]))
        pred_with_singular = pred.copy()
        pred_with_singular[np.where(pred == -1)] = np.arange(num_class_valid, num_class_valid + (pred == -1).sum()) # to assign -1 with new labels
        print("#cluster: {}".format(len(np.unique(pred_with_singular))))
        print("Cluster time: {:.2f} s".format(time.time() - start))
        
        
        exit()



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()