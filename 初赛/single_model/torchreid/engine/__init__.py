from __future__ import absolute_import
from __future__ import print_function

from .engine import Engine

from .image import ImageTripletEngine


show_oim_warning = False
if show_oim_warning:
    import warnings
    warnings.warn("OIMLoss also include in folder `losses`, but we do not open this in all model, only the models which "
                  "inherit from ResNet and ResNet_IBN_B class and embedding layer opened can use this, you can use:"
                  ">>> from torchreid import models"
                  ">>> models.show_avai_oim_loss_models()" 
                  "to show which models can be support. And for those supported model, oim loss is computed with "
                  "embedding_layer feature, as we know, when oim compute on embedding_layer, good performance can be "
                  "achieved evev only oim loss used! Thus you need to return embedding_layer in your model if you want to "
                  "support more model and we suggest you do not use only oim loss, use it with softmax jointly and evev use"
                  "those three loss maybe a good choice... so we use the latter and write it in engine.image.triplet.py, "
                  "note that: both image and video softmax.py are not extending ... and when you realized those function, "
                  "please delete this warning in engine.__init__.py")