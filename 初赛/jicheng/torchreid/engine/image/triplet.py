from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
#from torchreid.engine import engine
from torchreid.engine import engine_duo1

from torchreid import models
from torchreid.losses import CrossEntropyLoss, TripletLoss, OIMLoss, CenterLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics



class ImageTripletEngine(engine_duo1.Engine):

    def __init__(self, datamanager, model_1,model_2,model_3,model_4,model_5, optimizer, num_classes, margin=0.7, #0.3
                 weight_t=1, weight_x=1, scheduler=None, use_gpu=True,
                 label_smooth=True, use_oim=False, weight_o=1, use_center=False, center_lr=0.5, weight_c=0.0005):
        #super(ImageTripletEngine, self).__init__(datamanager,  model_1,model_2, optimizer, scheduler)   #, use_gpu
        super(ImageTripletEngine, self).__init__(datamanager, model_1,model_2,model_3,model_4,model_5,optimizer, scheduler)

        self.weight_t = weight_t
        self.weight_x = weight_x
        
        self.criterion_t = TripletLoss(margin=margin)
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
        self.weight_o = weight_o
        self.use_center = use_center
        self.weight_c = weight_c
        if self.use_oim == True:
            print("Only the following model can support OIM loss, please check your model or extend `oim` loss in "
                  "your model...")
            print(models.show_avai_oim_loss_models())

        self.criterion_o = OIMLoss(num_features=512, num_classes=num_classes,
                                   scalar=1.0, momentum=0.5, weight=None, size_average=True, use_gpu=self.use_gpu)

        if self.use_center == True:
            import warnings
            warnings.warn("Center loss using..., you can only use run, but not along test or resume... it is complex"
                          "to load params, so we drop out it...")
            self.criterion_c = CenterLoss(num_classes=num_classes, feat_dim=2048, use_gpu=self.use_gpu)  # resnet50: 2048  densenet121: 1024
            self.optimizer_center = torch.optim.SGD(self.criterion_c.parameters(), lr=center_lr)

    def train(self, epoch, max_epoch, trainloader, warm=None, fixbase_epoch=0, open_layers=None, print_freq=10, triplet_add_epoch=0, label_smooth_end_epoch=10):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
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
                if self.use_center:
                    loss_c = self._compute_loss(self.criterion_c, features, pids)
                    loss += self.weight_c * loss_c

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
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss_t=losses_t,
                      loss_x=losses_x,
                      acc=accs,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
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
