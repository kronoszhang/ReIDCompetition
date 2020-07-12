from __future__ import absolute_import
from __future__ import division

from .senet import HighDivModule, se_resnext101_32x4d
import torch
import torch.nn as nn

class senet_MHN(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super(senet_MHN, self).__init__()
        self.loss = loss
        self.base_model = se_resnext101_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs)
        
        self.parts = 4
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            setattr(self, name, HighDivModule(512, i+1))
    
    def featuremaps(self, x):
        x = self.base_model.layer0(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x_=[]
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            layer = getattr(self, name)
            x_.append(layer(x))
        x = torch.cat(x_, 0)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # x = self.base_model.featuremaps(x)
        return x

    def forward(self, x, return_featuremaps=False):
        f = self.featuremaps(x)
        if return_featuremaps:
            return f
            
        
        v = self.base_model.global_avgpool(f)
        v = v.view(v.size(0), -1)
        v1 = v
        
        if self.base_model.bnneck_layer is not None:
            v = self.base_model.bnneck_layer(v)
            v2 = v
        
        if self.base_model.fc is not None:
            v = self.base_model.fc(v)
            v3 = v
        
        if not self.training:
            if self.base_model.bnneck_test == 'before':
                return v1
            elif self.base_model.bnneck_test == 'after':
                return v2
            elif self.base_model.bnneck_test == 'fc':
                return v3
        
        y = self.base_model.classifier(v)
        
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v1
        elif self.loss == 'oim':
            # maybe not correct
            v3 = v1
            return y, v1, v3
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))