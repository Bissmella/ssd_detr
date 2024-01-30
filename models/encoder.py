import torch
import torch.nn as nn
from util.misc import inverse_sigmoid, _get_clones, _get_activation_fn
from timm.models.layers import trunc_normal_



class GlobalEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type='post_norm',
    ):
        super().__init__()

        self.norm_type = norm_type
        self.n_heads = n_heads
        self.img_attn = nn.MultiheadAttention(d_model, n_heads, dropout= dropout)
        self.dropout_img = nn.Dropout(dropout)
        self.norm_img = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self,
        src,
        src_pos_embed,
        prmpt = None,
        prmpt_mask = None,
        prmpt_pos_embed= None,
        img_mask = None
    ):
        
        src2 = self.norm_img(src)
        src2 = self.img_attn(
            self.with_pos_embed(src2, src_pos_embed).transpose(0, 1),
            self.with_pos_embed(prmpt, prmpt_pos_embed).transpose(0, 1),
            prmpt.transpose(0, 1),
            attn_mask = img_mask,
            key_padding_mask = prmpt_mask
        )[0].transpose(0, 1)
        src = src + self.dropout_img(src2)

        # ffn
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src2))))
        src = src + self.dropout4(src2)

        return src

    def forward_post(
        self,
        src,
        src_pos_embed,
        src_padding_mask=None,
        prmpt = None,
        prmpt_mask = None,
        prmpt_pos_embed= None,
    ):

        src2 = self.img_attn(
            src,
            self.with_pos_embed(prmpt, prmpt_pos_embed),
            prmpt,
            key_padding_mask = prmpt_mask.transpose(0, 1),
        )[0]
        src = src + self.dropout_img(src2)
        src = self.norm_img(src)

        # cross attention
    

        return src

    def forward(
        self,
        src,
        src_pos_embed,
        prmpt = None,
        prmpt_mask = None,
        prmpt_pos_embed= None,
        img_mask = None
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(src, src_pos_embed, prmpt, prmpt_mask, prmpt_pos_embed, img_mask)
        if self.norm_type == "post_norm":
            return self.forward_post(src, src_pos_embed, prmpt, prmpt_mask, prmpt_pos_embed, img_mask)


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        n_heads,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.n_heads = n_heads





    def _reset_parameters(self):

        # stolen from Swin Transformer
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, src, src_pos_embed, prompt, prompt_mask, prompt_pos, img_mask = None):

        for layer in self.layers:
            src = layer(
                        src,
                        src_pos_embed,
                        prmpt = prompt,
                        prmpt_mask = prompt_mask,
                        prmpt_pos_embed= prompt_pos,
                        img_mask = img_mask)
        
        return src
    
def build_global_encoder(args):
    encoder_layer = GlobalEncoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        n_heads=args.nheads,
        norm_type=args.norm_type,
    )
    encoder = GlobalEncoder(
        encoder_layer,
        num_layers=args.dec_layers,
        #return_intermediate=True,
        n_heads=args.nheads,
    )
    return encoder
