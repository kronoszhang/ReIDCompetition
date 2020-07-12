from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import copy
from PIL import Image
from zipfile import ZipFile
import os
import sys
import os.path as osp
import torchreid

def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
            pass
    return img


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid).
        val_query (list): contains tuples of (img_path(s), pid).
        val_gallery (list): contains tuples of (img_path(s), pid).
        query (list): contains tuples of (img_path(s), pid).
        gallery (list): contains tuples of (img_path(s)).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, val_query, val_gallery, query, gallery, transform=None, mode='train', verbose=True, **kwargs):
        self.train = train
        # split by ourself
        self.val_query = val_query  
        self.val_gallery = val_gallery
        # the provided dataset
        self.query = query
        self.gallery = gallery
        
        self.transform = transform
        self.mode = mode
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'val_query':
            self.data = self.val_query
        elif self.mode == 'val_gallery':
            self.data = self.val_gallery
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | val_query | val_gallery | query | gallery ]'.format(self.mode))

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        img_path, pid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, img_path

    def __len__(self):
        return len(self.data)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        for _, pid in data:
            pids.add(pid)
        return len(pids)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)

    def show_summary(self):
        """Shows dataset statistics."""
        num_train_pids = self.parse_data(self.train)
        num_val_query_pids = self.parse_data(self.val_query)
        num_val_gallery_pids = self.parse_data(self.val_gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images')
        print('  ----------------------------------------')
        print('  train    |  {:5d}  | {:8d}'.format(num_train_pids, len(self.train)))
        print('  Vquery   |  {:5d}  | {:8d}'.format(num_val_query_pids, len(self.val_query)))
        print('  Vgallery |  {:5d}  | {:8d}'.format(num_val_gallery_pids, len(self.val_gallery)))
        print('  query    | UNKNOWN | {:8d}'.format(len(self.query)))
        print('  gallery  | UNKNOWN | {:8d}'.format(len(self.gallery)))
        print('  ----------------------------------------')


class MyDataset(Dataset):
    """
        - identities: 4,768. (2272 train id + 500 val id + 1996 only_one_image id)
        - images: 20429 (train) + 1,348 (query) + 5,366(gallery).
        - cameras: None.
    """

    def __init__(self, root="project/ywchong/ywchong/CODE/zc/AIC/data", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))

        # unzip yourself, else messy code caused
        # train_path = osp.join(root, 'train.zip')
        # test_path = osp.join(root, 'test_a.zip')
        #
        # # Extract the file
        # exdir = osp.join(root, 'MyDataset')
        # if not osp.isdir(exdir):
        #     print("Extracting zip file...")
        #     with ZipFile(train_path) as z:
        #        z.extractall(path=root)
        #     with ZipFile(test_path) as z:
        #         z.extractall(path=root)  
        self.train_dir = osp.join(root, "trainset/trainset_remove_1_2_1000")  # change here to adjust the train set
        self.val_query_dir = osp.join(root, "valset/query")  # change here to adjust the val set query
        self.val_gallery_dir = osp.join(root, "valset/gallery")  # change here to adjust the val set gallery
        #self.query_dir = osp.join(root, "testset/query_a")  # true query
        #self.gallery_dir = osp.join(root, "testset/gallery_a")  # true gallery, no pid label
        
        # self.val_query_dir = osp.join("/data/reid/MSMT_V2_ZZ/", "query")  # change here to adjust the val set query
        # self.val_gallery_dir = osp.join("/data/reid/MSMT_V2_ZZ/", "bounding_box_test")  # change here to adjust the val set gallery
        # self.query_dir = osp.join("/data/reid/MSMT_V2_ZZ/", "query")  # change here to adjust the val set query
        # self.gallery_dir = osp.join("/data/reid/MSMT_V2_ZZ/", "bounding_box_test")  # change here to adjust the val set gallery
        self.query_dir = osp.join("/data/reid/NAIC/fu_b/", "query_b")  # true query
        self.gallery_dir = osp.join("/data/reid/NAIC/fu_b/", "gallery_b")  # true gallery, no pid label

        train = self.process_dir(self.train_dir, relabel=True)  # we remove some images, thus this must relabel, else the pid_value maybe exceed total class num
        val_query = self.process_dir(self.val_query_dir)
        val_gallery = self.process_dir(self.val_gallery_dir)
        query = self.process_dir(self.query_dir, no_pid=True)  # this pid exist, but maybe useless except multi-query, and best not modify this 
        gallery = self.process_dir(self.gallery_dir, no_pid=True)

        super(MyDataset, self).__init__(train, val_query, val_gallery, query, gallery, **kwargs)

    def process_dir(self, dir_path, no_pid=False, relabel=False):
        data = []
        pid_container = set()
        for image_name in os.listdir(dir_path):
            pid = int(image_name.split("_")[0]) if not no_pid else -1
            pid_container.add(pid)
    
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        for image_name in os.listdir(dir_path):
            pid = int(image_name.split("_")[0]) if not no_pid else -1  # when retrieve, use distance but not pid, thus random pid is given and do not effect the result
            img_path = osp.join(dir_path, image_name)
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid,))
        return data



class MyDataManager():
    def __init__(self, root="project/ywchong/ywchong/CODE/zc/AIC/data", height=256, width=128,
                 transforms='random_flip', norm_mean=None, norm_std=None, use_gpu=True, batch_size_train=32,
                 batch_size_test=32, workers=4, num_instances=4, train_sampler='', loss='softmax',):

        self.height = height
        self.width = width
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        super(MyDataManager, self).__init__()

        self.transform_tr, self.transform_te = torchreid.data.transforms.build_transforms(
            self.height, self.width, transforms=transforms,
            norm_mean=norm_mean, norm_std=norm_std
        )

        print('=> Loading train dataset')
        trainset = MyDataset(root, transform=self.transform_tr, mode='train')
        train_sampler = torchreid.data.sampler.build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances,
            loss=loss,
        )
        self.num_train_pids = trainset.parse_data(trainset.train)

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading val (self_test) dataset')
        self.valloader = {'val_query': None, 'val_gallery': None}
        self.valdataset = {'val_query': None, 'val_gallery': None}
        val_queryset = MyDataset(root, transform=self.transform_te, mode='val_query')
        self.valloader['val_query'] = torch.utils.data.DataLoader(
            val_queryset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False)
        val_galleryset = MyDataset(root, transform=self.transform_te, mode='val_gallery', verbose=False)
        self.valloader['val_gallery'] = torch.utils.data.DataLoader(
            val_galleryset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False)
        self.valdataset['val_query'] = val_queryset.query
        self.valdataset['val_gallery'] = val_galleryset.gallery
        
        
        print('=> Loading raw test dataset')
        self.testloader = {'query': None, 'gallery': None}
        self.testdataset = {'query': None, 'gallery': None}
        queryset = MyDataset(root, transform=self.transform_te, mode='query')
        self.testloader['query'] = torch.utils.data.DataLoader(
            queryset,
            batch_size=batch_size_test, 
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False)
        galleryset = MyDataset(root, transform=self.transform_te, mode='gallery', verbose=False)
        self.testloader['gallery'] = torch.utils.data.DataLoader(
            galleryset,
            batch_size=batch_size_test, 
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False)
        self.testdataset['query'] = queryset.query
        self.testdataset['gallery'] = galleryset.gallery
        print('\n')