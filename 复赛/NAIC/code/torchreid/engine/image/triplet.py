from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid import models
from torchreid.losses import CrossEntropyLoss, TripletLoss, OIMLoss, CenterLoss, Local_TripletLoss, local_loss, RankedLoss, FocalLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics



class ImageTripletEngine(engine.Engine):

    def __init__(self, datamanager, model, optimizer, num_classes, margin=0.3,
                 weight_t=1, weight_x=1, scheduler=None, use_gpu=True,
                 label_smooth=True, use_oim=False, weight_o=1, use_center=False, center_lr=0.5, weight_c=0.0005, oim_dims=512, center_dims=2048, use_ranked_loss=False, weight_r=1, use_focal=False, weight_f=1):
        super(ImageTripletEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_f = weight_f
        
        
        self.criterion_r = RankedLoss(margin=1.3, alpha=2.0, tval=1.0)
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_f = FocalLoss(gamma=0, alpha=None, size_average=True)  
        self.criterion_local_t = Local_TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=num_classes,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        
        self.criterion_x_no_smooth = CrossEntropyLoss(
            num_classes=num_classes,
            use_gpu=self.use_gpu,
            label_smooth=False
        )
        
        self.use_oim = use_oim
        self.use_focal = use_focal
        self.weight_o = weight_o
        self.use_center = use_center
        self.weight_c = weight_c
        self.use_ranked_loss = use_ranked_loss
        self.weight_r = weight_r
        if self.use_oim == True:
            print("Only the following model can support OIM loss, please check your model or extend `oim` loss in "
                  "your model...")
            print(models.show_avai_oim_loss_models())

        self.criterion_o = OIMLoss(num_features=oim_dims, num_classes=num_classes,
                                   scalar=1.0, momentum=0.5, weight=None, size_average=True, use_gpu=self.use_gpu)

        if self.use_center == True:
            import warnings
            warnings.warn("Center loss using..., you can only use run, but not along test or resume... it is complex"
                          "to load params, so we drop out it...")
            self.criterion_c = CenterLoss(num_classes=num_classes, feat_dim=center_dims, use_gpu=self.use_gpu)  # resnet50: 2048  densenet121: 1024 pcb_p6 based resnet50: 2048*6=12288
            self.optimizer_center = torch.optim.SGD(self.criterion_c.parameters(), lr=center_lr)

    def train(self, epoch, max_epoch, trainloader, warm=None, fixbase_epoch=0, open_layers=None, print_freq=10, triplet_add_epoch=0, label_smooth_end_epoch=10):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        losses_r = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if warm is not None:
            warm_epoch, warm_up, warm_iteration = warm['warm_epoch'],  warm['warm_up'], warm['warm_iteration']

        self.model.train()
        if (epoch+1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            self.optimizer.zero_grad()
            if self.use_center:
                self.optimizer_center.zero_grad()
            if not self.use_oim:
                outputs, features = self.model(imgs)
                # outputs, features, pcb_outputs, pcb_features, local_outputs, local_features = self.model(imgs)
                # outputs_ = self.model(imgs)
                # outputs, t1, t2, t3, l1, l2, l3 = outputs_[0:7]
                
                if (epoch+1) >= label_smooth_end_epoch: 
                    if (epoch+1) == label_smooth_end_epoch and batch_idx==0:
                        print("from {} epoch, label smooth would remove...".format(label_smooth_end_epoch))
                    loss_x = self._compute_loss(self.criterion_x_no_smooth, outputs, pids)
                    #loss_x1 = self._compute_loss(self.criterion_x_no_smooth, l1, pids)
                    #loss_x2 = self._compute_loss(self.criterion_x_no_smooth, l2, pids)
                    #loss_x3 = self._compute_loss(self.criterion_x_no_smooth, l3, pids)
                    #loss_x1 = self._compute_loss(self.criterion_x_no_smooth, pcb_outputs, pids)
                    #loss_x2 = self._compute_loss(self.criterion_x_no_smooth, local_outputs, pids)
                else:
                    loss_x = self._compute_loss(self.criterion_x, outputs, pids)
                    #loss_x1 = self._compute_loss(self.criterion_x, l1, pids)
                    #loss_x2 = self._compute_loss(self.criterion_x, l2, pids)
                    #loss_x3 = self._compute_loss(self.criterion_x, l3, pids)
                    #loss_x1 = self._compute_loss(self.criterion_x, pcb_outputs, pids)
                    #loss_x2 = self._compute_loss(self.criterion_x, local_outputs, pids)
                # loss = self.weight_x * (loss_x1 + loss_x2 + loss_x3)# + 0*loss_x1 + 0*loss_x2)
                loss = self.weight_x * loss_x
                if self.use_ranked_loss:
                    loss_r = self._compute_loss(self.criterion_r, outputs, pids)
                    losses_r.update(loss_r.item(), pids.size(0))
                    loss += self.weight_r * loss_r
                if self.use_focal:
                    loss_f = self._compute_loss(self.criterion_f, outputs, pids)
                    loss += self.weight_f * loss_f
                if (epoch+1) >= triplet_add_epoch:
                    if (epoch+1) == triplet_add_epoch and batch_idx==0:
                        print("from {} epoch, triplet loss added...".format(triplet_add_epoch))
                    loss_t = self._compute_loss(self.criterion_t, features, pids)
                    #loss_t1 = self._compute_loss(self.criterion_t, t1, pids)
                    #loss_t2 = self._compute_loss(self.criterion_t, t2, pids)
                    #loss_t3 = self._compute_loss(self.criterion_t, t3, pids)
                    #loss_t1 = self._compute_loss(self.criterion_t, pcb_features, pids)
                    #loss_t2 = local_loss(self.criterion_local_t, local_features, pids, normalize_feature=True)
                    #loss += self.weight_t * (loss_t1 + loss_t2 + loss_t3)# + 0*loss_t1 + 0*loss_t2)
                    loss += self.weight_t * loss_t
                if self.use_center:
                    loss_c = self._compute_loss(self.criterion_c, features, pids)
                    #loss_c1 = self._compute_loss(self.criterion_c, t1, pids)
                    #loss_c2 = self._compute_loss(self.criterion_c, t2, pids)
                    #loss_c3 = self._compute_loss(self.criterion_c, t3, pids)
                    # each feature must use one center loss and center optimizer
                    # loss_c1 = self._compute_loss(self.criterion_c, pcb_features, pids)  # pcb not use center loss for convience
                    # local_features = local_features.permute(0, 2, 1)
                    # local_features_view = local_features.view(local_features.size(0), -1)
                    #loss_c2 = self._compute_loss(self.criterion_c, local_features_view, pids)
                    #loss += self.weight_c * (loss_c1 + loss_c2 + loss_c3)# + loss_c2)
                    loss += self.weight_c * loss_c
                #loss_vis = (loss_x.item(), loss_x1.item(), loss_x2.item(), loss_t.item(), loss_t1.item(), loss_t2.item(), loss_c.item(), loss_c2.item(), loss.item())
                # loss_vis = (loss_x.item(), 0, 0, loss_t.item(), 0, 0, loss_c.item(), 0, loss.item())
                #loss_vis = (loss_x1.item(), loss_x2.item(), loss_x3.item(), loss_t1.item(), loss_t2.item(), loss_t3.item(), loss_c1.item(), loss_c2.item(), loss_c3.item(), loss.item())
                
            else:
                outputs, features, embedding_feature = self.model(imgs)
                if (epoch+1) >= label_smooth_end_epoch: 
                    if (epoch+1) == label_smooth_end_epoch and batch_idx==0:
                        print("from {} epoch, label smooth would remove...".format(label_smooth_end_epoch))
                    loss_x = self._compute_loss(self.criterion_x_no_smooth, outputs, pids)
                else:
                    loss_x = self._compute_loss(self.criterion_x, outputs, pids)
                loss = self.weight_x * loss_x
                if (epoch+1) >= triplet_add_epoch:
                    if (epoch+1) == triplet_add_epoch and batch_idx==0:
                        print("from {} epoch, triplet loss added...".format(triplet_add_epoch))
                    loss_t = self._compute_loss(self.criterion_t, features, pids)
                    loss += self.weight_t * loss_t
                loss_o = self._compute_loss(self.criterion_o, embedding_feature, pids)[0]
                loss += self.weight_o * loss_o
                if self.use_center:
                    loss_c = self._compute_loss(self.criterion_c, features, pids)
                    loss += self.weight_c * loss_c

            # for lr warm up
            if (epoch + 1) < warm_epoch and warm is not None:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up
                warm['warm_up'] = warm_up

            loss.backward()
            self.optimizer.step()

            if self.use_center:
                for param in self.criterion_c.parameters():
                    param.grad.data *= (1. / self.weight_c)
                self.optimizer_center.step()

            batch_time.update(time.time() - end)
            
            if (epoch+1) >= triplet_add_epoch:
                losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                      'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                      'Loss_r {loss_r.val:.4f} ({loss_r.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}\t'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss_t=losses_t,
                      loss_x=losses_x,
                      loss_r=losses_r,
                      acc=accs,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str,
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                self.writer.add_scalar('Train/Loss_t', losses_t.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x', losses_x.avg, n_iter)
                self.writer.add_scalar('Train/Acc', accs.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        return warm
