import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder

class Model(nn.Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, eval=False, img_mask=None):
        if self.args.dataset_name == 'iu_xray':
            att_feats_0 = self.visual_extractor(images[:, 0])
            att_feats_1 = self.visual_extractor(images[:, 1])

            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

            sentences, log_probs = self.encoder_decoder(att_feats, mode='sample')
            return sentences, log_probs
            
        else:
            bs, pic_len, c, w, h = images.shape
            images = images.reshape(-1, c, w, h)
            att_feats = self.visual_extractor(images)  # (bs*4, 3 ,224, 224)

            # generating masks
            patch_num = att_feats.shape[1]
            img_padding_mask = img_mask.unsqueeze(-1).repeat((1, 1, patch_num))
            att_feats = att_feats.reshape(bs, pic_len*patch_num, -1)
            img_padding_mask = img_padding_mask.reshape(bs, -1)

            sentences, log_probs = self.encoder_decoder(att_feats, mode='sample', att_masks=img_padding_mask, eval=eval)
            return sentences, log_probs
            
            


