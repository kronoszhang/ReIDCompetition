from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results']

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