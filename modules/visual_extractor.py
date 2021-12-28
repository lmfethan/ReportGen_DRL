import torch
import torch.nn as nn
import torchvision.models as models
from modules import attention

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        model.layer3.add_module('att_layer3', attention.SimAM())
        model.layer4.add_module('att_layer4', attention.SimAM())

        modules = list(model.children())[:-2]

        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats



