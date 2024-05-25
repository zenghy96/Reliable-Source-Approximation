""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class EvidentialUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(EvidentialUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.transform_v = (OutConv(64, n_classes))
        self.transform_alpha = (OutConv(64, n_classes))
        self.transform_beta = (OutConv(64, n_classes))
        self._ev_dec_v_max = 20
        self._ev_dec_alpha_max = 20
        self._ev_dec_beta_min = 0.2

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        # --- evidence ---
        logv = self.transform_v(x)
        logalpha = self.transform_alpha(x)
        logbeta = self.transform_beta(x)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)

        alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha.shape).to(alpha.device)
        alpha = torch.min(alpha, alpha_thr)
        v_thr = self._ev_dec_v_max * torch.ones(v.shape).to(v.device)
        v = torch.min(v, v_thr)
        beta_min = self._ev_dec_beta_min * torch.ones(beta.shape).to(beta.device)
        beta = beta + beta_min

        return logits, v, alpha, beta
