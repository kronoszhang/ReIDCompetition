from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import math
import os.path as osp
import time
import datetime
import numpy as np
import cv2
import scipy.io

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.autograd import Variable
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import torchreid
from torchreid.utils import AverageMeter, visualize_ranked_results, save_checkpoint, re_ranking, mkdir_if_missing
from torchreid.losses import DeepSupervision
from torchreid import metrics


GRID_SPACING = 10


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_gpu=True):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None

        # check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def run(self, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, test_only=False, print_freq=10,
            dist_metric='euclidean', normalize_feature=False,
            use_metric_cuhk03=False, ranks=[1, 5, 10, 20], rerank=False,
            warm_epoch=10,  # We warm up in first 10 epoch
            warm_up=0.1,  # We start from the 0.1*lrRate
            label_smooth_end_epoch=10,
            ):
    
        trainloader, valloader, testloader, testdataset = self.datamanager.trainloader, self.datamanager.valloader, self.datamanager.testloader, self.datamanager.testdataset
        
        # print(len(trainloader), len(valloader), len(testloader))
        if test_only:
            self.test(
                0,
                valloader,
                testloader,
                testdataset,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        if self.writer is None:
            # for tensorboard verson difference, you should choose one of the following 
            #self.writer = SummaryWriter(logdir=save_dir)
            self.writer = SummaryWriter(log_dir=save_dir)


        time_start = time.time()
        print('=> Start training')

        # for lr warm up
        # warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * warm_epoch  # this is in reid-baseline of ZZD
        # trainloader = self.datamanager.trainloader
        warm_iteration = len(trainloader) * warm_epoch  # warm up in first 10 epoch, same effect
        warm = {'warm_epoch': warm_epoch,
                'warm_up': warm_up,
                'warm_iteration': warm_iteration,}
        for epoch in range(start_epoch, max_epoch):
            warm = self.train(epoch, max_epoch, trainloader, warm, fixbase_epoch, open_layers, print_freq, label_smooth_end_epoch=label_smooth_end_epoch) ###########################
            # print(warm['warm_up'])
            
            if (epoch+1)>=start_eval and eval_freq>0 and (epoch+1)%eval_freq==0 and (epoch+1)!=max_epoch:
                rank1 = self.test(
                    epoch,
                    valloader,
                    testloader=None,
                    testdataset=None,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self._save_checkpoint(epoch, rank1, save_dir)

        if max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                epoch,
                valloader,
                testloader,
                testdataset,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is None:
            self.writer.close()

    def train(self):
        r"""Performs training on source datasets for one epoch.

        This will be called every epoch in ``run()``, e.g.

        .. code-block:: python
            
            for epoch in range(start_epoch, max_epoch):
                self.train(some_arguments)

        .. note::
            
            This must be implemented in subclasses.
        """
        raise NotImplementedError

    def test(self, epoch, valloader, testloader, testdataset, dist_metric='euclidean', normalize_feature=False,
             save_dir='', use_metric_cuhk03=False,
             ranks=[1, 5, 10, 20], rerank=False):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        print('##### Evaluating Val Dataset...  #####')
        valqueryloader = valloader['val_query']
        valgalleryloader = valloader['val_gallery']
        rank1 = self._evaluate(
            epoch,
            queryloader=valqueryloader,
            galleryloader=valgalleryloader,
            testdataset=None,
            dist_metric=dist_metric,
            normalize_feature=normalize_feature,
            save_dir=save_dir,
            use_metric_cuhk03=use_metric_cuhk03,
            ranks=ranks,
            rerank=rerank
        )
        if testloader is not None:
            print('##### Retrieve Test Dataset...  #####')
            queryloader = testloader['query']
            galleryloader = testloader['gallery']
            self._evaluate(
                epoch,
                queryloader=queryloader,
                galleryloader=galleryloader,
                testdataset=testdataset,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                return_json=True
            )
        
        return rank1

    def fliplr(self, img):
        """flip horizontal"""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    @torch.no_grad()
    def _evaluate(self, epoch, queryloader=None, galleryloader=None, testdataset=None,
                  dist_metric='euclidean', normalize_feature=False, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                  rerank=False, multi_scale_interpolate_mode='bilinear', multi_scale=(1,), return_json=False):  # multi_scale=(1, 1.1, 1.2,) 

        batch_time = AverageMeter()

        ms = []
        for scale in multi_scale:
            scale_f = float(scale)
            ms.append(math.sqrt(scale_f))

        print('Extracting features from query set ...')
        imgs, pids = self._parse_data_for_eval(next(iter(queryloader)))
        if self.use_gpu:
            imgs = imgs.cuda()
        feature_test = self._extract_features(imgs)  # bs not suitable for last batch size
        qf, q_pids = [], []  # query features, query person IDs 
        for batch_idx, data in enumerate(queryloader):
            imgs, pids = self._parse_data_for_eval(data)
            end = time.time()
            bs, c, h, w = imgs.shape
            features = torch.zeros_like(feature_test)[0:bs]
            for index in range(2):
                # index=0 for flip image and index=1 for raw image
                if index == 0:
                    imgs = Variable(self.fliplr(imgs))
                if self.use_gpu:
                    imgs = imgs.cuda()
                    features = features.cuda()
                for scale in ms:
                    if scale != 1:
                        imgs = nn.functional.interpolate(imgs, scale_factor=scale, mode=multi_scale_interpolate_mode,
                                                         align_corners=False)
                    features_ = self._extract_features(imgs)
                    features += features_

            features = features / 2 / len(ms)  # use mean value of raw image and flip image as the final feature
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids = [], [] # gallery features, gallery person IDs
        end = time.time()
        for batch_idx, data in enumerate(galleryloader):
            imgs, pids = self._parse_data_for_eval(data)
            end = time.time()
            bs, c, h, w = imgs.shape
            features = torch.zeros_like(feature_test)[0:bs]
            for index in range(2):
                # index=0 for flip image and index=1 for raw image
                if index == 0:
                    imgs = Variable(self.fliplr(imgs))
                if self.use_gpu:
                    imgs = imgs.cuda()
                    features = features.cuda()
                for scale in ms:
                    if scale != 1:
                        imgs = nn.functional.interpolate(imgs, scale_factor=scale, mode=multi_scale_interpolate_mode,
                                                         align_corners=False)
                    features_ = self._extract_features(imgs)
                    features += features_
            features = features / 2 / len(ms)  # use mean value of raw image and flip image as the final feature
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        # Save to Matlab for check
        result = {'gallery_f': gf.numpy(), 'gallery_label': g_pids,
                  'query_f': qf.numpy(), 'query_label': q_pids,}
        scipy.io.savemat('pytorch_result.mat', result)

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()



        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        
        if not return_json:
            # evaluate result on valset
            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                use_metric_cuhk03=use_metric_cuhk03
            )
    
            print('** Results **')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))
    
            return cmc[0]
        else:
            visualize_ranked_results(
                distmat,
                testdataset,
                save_dir=save_dir,
            )
        
    

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)
        with open('./log/baseline2/model.pth.tar-{}'.format(epoch + 1),'rb') as f,open('../fwq/My Drive/Colab/Game_code/experiment/storage/model.pth.tar-{}'.format(epoch + 1),"wb") as fw:
            fw.write(f.read())
