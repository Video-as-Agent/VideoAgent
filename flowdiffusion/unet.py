from guided_diffusion.guided_diffusion.unet import UNetModel
from torch import nn
import torch
from einops import repeat, rearrange


class UnetBridge(nn.Module):
    def __init__(self):
        super(UnetBridge, self).__init__()

        self.unet = UNetModel(
            image_size=(48, 64),
            in_channels=6,
            model_channels=160,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, t, task_embed=None, **kwargs):
        # f = x.shape[1] // 3 - 1 
        # print("f",f)
        # ## f 6
        # print("UNet x shape", x.shape)
        # ## UNet x shape torch.Size([32, 21, 48, 64])
        # x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        # print("UNet x_cond shape", x_cond.shape)
        # ## UNet x_cond shape torch.Size([32, 3, 6, 48, 64])
        # x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        # x = torch.cat([x, x_cond], dim=1)
        # print("UNet x shape", x.shape)
        # ## UNet x shape torch.Size([32, 6, 6, 48, 64])
        out = self.unet(x, t, task_embed, **kwargs)
        return out

# class UnetMW(nn.Module):
#     def __init__(self, freeze_blocks=21):
#         super(UnetMW, self).__init__()
#         self.unet = UNetModel(
#             image_size=(128, 128),
#             in_channels=6,
#             model_channels=128,
#             out_channels=3,
#             num_res_blocks=2,
#             attention_resolutions=(8, 16),
#             dropout=0,
#             channel_mult=(1, 2, 3, 4, 5),
#             conv_resample=True,
#             dims=3,
#             num_classes=None,
#             task_tokens=True,
#             task_token_channels=512,
#             use_checkpoint=False,
#             use_fp16=False,
#             num_head_channels=32,
#         )
#         self.freeze_initial_blocks(freeze_blocks)

#     def freeze_initial_blocks(self, num_blocks):
#         if num_blocks <= 0:
#             return

#         blocks_frozen = 0
#         total_blocks = len(self.unet.input_blocks) + 1 + len(self.unet.output_blocks) + 1

#         def freeze_params(module):
#             for param in module.parameters():
#                 param.requires_grad = False

#         # Freeze input blocks
#         for block in self.unet.input_blocks:
#             if blocks_frozen >= num_blocks:
#                 break
#             freeze_params(block)
#             blocks_frozen += 1

#         # Freeze middle block if necessary
#         if blocks_frozen < num_blocks:
#             freeze_params(self.unet.middle_block)
#             blocks_frozen += 1

#         # Freeze output blocks if necessary
#         for block in self.unet.output_blocks:
#             if blocks_frozen >= num_blocks:
#                 break
#             freeze_params(block)
#             blocks_frozen += 1

#         # Freeze final output layer if necessary
#         if blocks_frozen < num_blocks and blocks_frozen < total_blocks:
#             freeze_params(self.unet.out)
#             blocks_frozen += 1

#         print(f"Froze the first {blocks_frozen} blocks of the U-Net out of {total_blocks} total blocks")

#         # Ensure at least one block is trainable
#         if blocks_frozen >= total_blocks:
#             print("Warning: All blocks were frozen. Unfreezing the last block to ensure trainability.")
#             if len(self.unet.output_blocks) > 0:
#                 for param in self.unet.output_blocks[-1].parameters():
#                     param.requires_grad = True
#             else:
#                 for param in self.unet.out.parameters():
#                     param.requires_grad = True

#     def unfreeze_all_layers(self):
#         for param in self.unet.parameters():
#             param.requires_grad = True
#         print("Unfroze all layers of the U-Net")

#     def forward(self, x, t, task_embed=None, **kwargs):
#         out = self.unet(x, t, task_embed, **kwargs)
#         return out

class UnetMW(nn.Module):
    def __init__(self):
        super(UnetMW, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, t, task_embed=None, **kwargs):
        # f = x.shape[1] // 3 - 1 
        # x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        # x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        # x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return out
      
class UnetMW_flow(nn.Module):
    def __init__(self):
        super(UnetMW_flow, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=5,
            model_channels=128,
            out_channels=2,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, t, task_embed=None, **kwargs):
        f = x.shape[1] // 2 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', f=f) 
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class UnetThor(nn.Module):
    def __init__(self):
        super(UnetThor, self).__init__()

        self.unet = UNetModel(
            image_size=(64, 64),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, t, task_embed=None, **kwargs):
        #f = x.shape[1] // 3 - 1 
        #print(x.shape)
        #x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        #x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        #x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return out # rearrange(out, 'b c f h w -> b (f c) h w')
    

