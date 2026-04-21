import torch
import torch.nn as nn

def get_time_emb(time_steps, dim):
    assert dim % 2 == 0, "Time embedding dimension must be even"
    factor = 10000 ** (torch.arange(0, dim // 2, dtype=torch.float32, device=time_steps.device) // (dim // 2))

    temb = time_steps[:, None].repeat(1, dim//2) / factor
    temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim = -1)
    return temb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.res_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.temb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_channels)
        )
        self.res_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.res_inp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x, temb):
        out = x

        #resnet
        res_inp = out
        out = self.res_conv_first(out)
        out = out + self.temb_layer(temb)[:, :, None, None]
        out = self.res_conv_second(out)
        out = out + self.res_inp_conv(res_inp)

        b, c, h, w = out.shape
        in_attn = out.reshape(b, c, h*w)
        in_attn = self.attn_norm(in_attn).permute(0, 2, 1)
        out_attn, _ = self.attn(in_attn, in_attn, in_attn)
        out_attn = out_attn.permute(0, 2, 1).reshape(b, c, h, w) 
        out = out + out_attn

        out = self.down_sample_conv(out)
        return out
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, num_heads):
        super().__init__()
        self.res_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        ])
        self.temb_layer = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_dim, out_channels)
            )
        ])
        self.res_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        ])
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.res_inp_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1), 
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])
    
    def forward(self, x, temb):
        out = x
        res_inp = out
        out = self.res_conv_first[0](out)
        out = out + self.temb_layer[0](temb)[:, :, None, None]
        out = self.res_conv_second[0](out)
        out = out + self.res_inp_conv[0](res_inp)

        b, c, h, w = out.shape
        in_attn = out.reshape(b, c, h*w)
        in_attn = self.attn_norm(in_attn).permute(0, 2, 1)
        out_attn, _ = self.attn(in_attn, in_attn, in_attn)
        out_attn = out_attn.permute(0, 2, 1).reshape(b, c, h, w)
        out = out + out_attn

        res_inp = out

        out = self.res_conv_first[1](out)
        out = out + self.temb_layer[1](temb)[:, :, None, None]
        out = self.res_conv_second[1](out)
        out = out + self.res_inp_conv[1](res_inp)

        return out
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.res_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.temb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_channels)
        )

        self.res_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.res_inp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up_sample_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity()
        
        def forward(self, x, out_down, temb):
            x = self.up_sample_conv(x)
            x = torch.cat([x, out_down], dim=1)

            out = x
            #resnet
            res_inp = out
            out = self.res_conv_first(out)
            out = out + self.temb_layer(temb)[:, :, None, None]
            out = self.res_conv_second(out)
            out = out + self.res_inp_conv(res_inp)

            b, c, h, w = out.shape
            in_attn = out.reshape(b, c, h*w)
            in_attn = self.attn_norm(in_attn).permute(0, 2, 1)
            out_attn, _ = self.attn(in_attn, in_attn, in_attn)
            out_attn = out_attn.permute(0, 2, 1).reshape(b, c, h, w)
            out = out + out_attn

            return out
        
        
