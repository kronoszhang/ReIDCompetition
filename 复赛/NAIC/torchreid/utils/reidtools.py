from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results', 'visualize_ranked_results_and_show_rank_list']

import numpy as np
import os
import os.path as osp
import shutil
import cv2

from .tools import mkdir_if_missing  # write_json exist in this file but this need utf-8 format, so we re-write this fuunction
from collections import OrderedDict



def visualize_ranked_results(distmat, testdataset, save_dir='', topk=200):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    import json
    def read_json(fpath):
        """Reads json file from a path."""
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj

    def write_json(obj, fpath):
        """Writes to a json file."""
        mkdir_if_missing(osp.dirname(fpath))
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, separators=(',', ': '))
            #json.dumps(obj, f, indent=4, separators=(',', ': '), ensure_ascii=False, encoding='utf-8')
            
            
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = testdataset['query'], testdataset['gallery']
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    result = OrderedDict()
    #print("======================")
    for q_idx in range(num_q):
        qimg_path, qpid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        #print("Deal Query Image: ", qimg_path)
        gimg_list = []
        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid = gallery[g_idx]
            gimg_list.append(gimg_path.split("/")[-1])
            rank_idx += 1
            if rank_idx > topk:
                break
        result[qimg_path.split("/")[-1]] = gimg_list
        
    """result = OrderedDict()
    # print("======================")
    for q_idx in range(num_q):
        qimg_path, qpid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        # print("Deal Query Image: ", qimg_path)
        rank_idx = 0
        new_list = []
        new_list.append(indices[q_idx, :][0])
        rank_idx += 1
        for g_idx in indices[-199:]:
            gimg_path, gpid = gallery[g_idx]
            new_list.append(gimg_path.split("/")[-1])
            rank_idx += 1
        assert rank_idx == 200
        result[qimg_path.split("/")[-1]] = new_list"""
    write_json(result, osp.join(save_dir, 'retrieve_result.json'))
    print('Done. Json file have been saved to "{}" ...'.format(save_dir))
    
    
    

# no use of NAIC dataset, because we have no label    
GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results_and_show_rank_list(distmat, testdataset, width=128, height=256, save_dir='', topk=10):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    
    import json
    def read_json(fpath):
        """Reads json file from a path."""
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj

    def write_json(obj, fpath):
        """Writes to a json file."""
        mkdir_if_missing(osp.dirname(fpath))
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, separators=(',', ': '))
            #json.dumps(obj, f, indent=4, separators=(',', ': '), ensure_ascii=False, encoding='utf-8')
            
            
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = testdataset['query'], testdataset['gallery']
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)
    
    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)



    result = OrderedDict()
    #print("======================")
    for q_idx in range(num_q):
        qimg_path, qpid = query[q_idx]
        # qimg_path, qpid, qcamid = query[q_idx] # normally we need this, but NAIC need not cross creama 
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        qimg = cv2.imread(qimg_path)
        qimg = cv2.resize(qimg, (width, height))
        qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # resize twice to ensure that the border width is consistent across images
        qimg = cv2.resize(qimg, (width, height))
        num_cols = topk + 1
        grid_img = 255 * np.ones((height, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
        grid_img[:, :width, :] = qimg
        
        # print("Deal Query Image: ", qimg_path)
        gimg_list = []
        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            # gimg_path, gpid, gcamid = gallery[g_idx] # normally we need this, but NAIC need not cross creama 
            gimg_path, gpid = gallery[g_idx]
            # invalid = (qpid == gpid) & (qcamid == gcamid) # normally we need this, but NAIC need not cross creama 
            invalid = False
            
            if not invalid:
                # for showwing rank list
                matched = gpid==qpid # we have no label for NAIC dataset, so need not this
                # matched = True # set this for NAIC dataset
                border_color = GREEN if matched else RED
                gimg = cv2.imread(gimg_path)
                gimg = cv2.resize(gimg, (width, height))
                gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                gimg = cv2.resize(gimg, (width, height))
                start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                end = (rank_idx+1)*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                grid_img[:, start: end, :] = gimg
               
                # for writing json file
                gimg_list.append(gimg_path.split("/")[-1])
                
                rank_idx += 1
                if rank_idx > topk:
                    break

        # for writing json file    
        result[qimg_path.split("/")[-1]] = gimg_list
        
        # save rank list images
        imname = osp.basename(osp.splitext(qimg_path_name)[0])
        cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))
        
    """result = OrderedDict()
    # print("======================")
    for q_idx in range(num_q):
        qimg_path, qpid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        # print("Deal Query Image: ", qimg_path)
        rank_idx = 0
        new_list = []
        new_list.append(indices[q_idx, :][0])
        rank_idx += 1
        for g_idx in indices[-199:]:
            gimg_path, gpid = gallery[g_idx]
            new_list.append(gimg_path.split("/")[-1])
            rank_idx += 1
        assert rank_idx == 200
        result[qimg_path.split("/")[-1]] = new_list"""
    write_json(result, osp.join(save_dir, 'retrieve_result.json'))
    print('Done. Json file have been saved to "{}" ...'.format(save_dir))