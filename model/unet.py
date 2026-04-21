import torch
import torch.nn as nn
from .blocks import DownBlock, UpBlock, get_time_emb

class UNet(nn.Module):
    def __init__(self, im_channels):
        super().__init__()
        self.down_ch = [32, 64, 128, 256]
        self.mid_ch = [256, 256, 128]
        self.up_ch = [256, 128, 64, 32]
        self.down_sample = [True, True, False]

        self.t_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim) 
        )

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_ch)-1):
            self.downs.append(DownBlock(self.down_ch[i], self.down_ch[i+1], self.time_emb_dim, self.down_sample[i], num_heads=4))   

        self.mid = nn.ModuleList([])
        for i in range(len(self.mid_ch)-1):
            self.mid.append(DownBlock(self.mid_ch[i], self.mid_ch[i+1], self.time_emb_dim, False, num_heads=4))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.up_ch)-1)):
            self.ups.append(UpBlock(self.down_ch[i]*2, self.up_ch[i-1] if i != 0 else 16, 
                                    self.time_emb_dim, self.down_sample[i], num_heads=4))
            
        self.conv_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        out = self.conv_in(x)
        temb = get_time_emb(t, self.time_emb_dim)
        temb = self.t_proj(temb)
        down_outs= []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, temb)
        for mid in self.mid:
            out = mid(out, temb)
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, temb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out
