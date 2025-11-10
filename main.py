import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

# Import necessary modules from provided files
# Assuming the following are available: VTTSAT.py, Transformer_Enc.py, Attention.py, Embed.py
# We'll integrate relevant parts

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4, kernel_size=3):
        super(SEBasicBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Lightweight VariableTemporal Self-Attention (reduced heads and dims for efficiency)
class LightweightVTSA(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.1):  # Reduced heads and dim_head for lightweight
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn=False):
        b, n, d = x.shape  # Assume input is b, seq, dim
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), weights

# Attention-based Denoising Module (inspired by DeepECG-Net and searches)
class AttentionDenoise(nn.Module):
    def __init__(self, input_channels=1, output_channels=32, num_heads=2):  # Lightweight with fewer heads
        super().__init__()
        # Proyección inicial si input_channels != output_channels
        if input_channels != output_channels:
            self.input_proj = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        else:
            self.input_proj = nn.Identity()
        
        self.mha = nn.MultiheadAttention(embed_dim=output_channels, num_heads=num_heads, dropout=0.1, batch_first=False)
        self.norm1 = nn.LayerNorm(output_channels)
        self.norm2 = nn.LayerNorm(output_channels)
        self.ff = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_channels, output_channels)
        )

    def forward(self, x):
        # x: b, c, l (1D signal, but we treat length as seq)
        x = self.input_proj(x)  # Proyectar a output_channels si es necesario
        # Reordenar para MultiheadAttention: (seq_len, batch, channels)
        x = x.permute(2, 0, 1)  # l, b, c
        # Self-attention
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        # Volver a b, c, l
        return x.permute(1, 2, 0)  # back to b, c, l

# Main Model: TF_MorphoTransNet
class TF_MorphoTransNet(nn.Module):
    def __init__(self, configs, hparams):
        super(TF_MorphoTransNet, self).__init__()

        # Denoising Module (lightweight attention-based)
        # Input es 1 canal (ECG), output es mid_channels para compatibilidad
        self.denoise = AttentionDenoise(input_channels=configs.input_channels, 
                                       output_channels=configs.mid_channels, 
                                       num_heads=2)

        # STFT Parameters for Time-Frequency Transform (using torch.stft for differentiability)
        self.stft_n_fft = 256
        self.stft_hop_length = 16
        self.stft_win_length = 256

        # TF-MG-MSC: 2D Convs for morphology-guided multi-scale (reduced channels for lightweight)
        filter_sizes = [(3, 3), (5, 5), (7, 7)]  # Narrow for QRS (high-freq), medium for P/T, wide for rhythm
        self.out_channels = 16  # Reduced from 32 for efficiency (guardado como atributo)
        
        # Calcular padding para mantener dimensiones (compatible con versiones antiguas de PyTorch)
        def get_padding(kernel_size):
            return kernel_size // 2
        
        self.conv2d_1 = nn.Conv2d(1, self.out_channels, kernel_size=filter_sizes[0], 
                                  padding=(get_padding(filter_sizes[0][0]), get_padding(filter_sizes[0][1])))
        self.conv2d_2 = nn.Conv2d(1, self.out_channels, kernel_size=filter_sizes[1], 
                                  padding=(get_padding(filter_sizes[1][0]), get_padding(filter_sizes[1][1])))
        self.conv2d_3 = nn.Conv2d(1, self.out_channels, kernel_size=filter_sizes[2], 
                                  padding=(get_padding(filter_sizes[2][0]), get_padding(filter_sizes[2][1])))
        self.bn2d = nn.BatchNorm2d(self.out_channels * 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(configs.dropout)

        # Dynamic Scale Fusion (DSF) with Gated Attention (lightweight gating)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * 3, 3, kernel_size=1),
            nn.Sigmoid()
        )

        # Variable-Temporal Self-Attention (lightweight version)
        # Actualizado: 8 heads como se menciona en el pipeline (pero reducido a 4 para eficiencia)
        self.vt_sa = LightweightVTSA(dim=self.out_channels, heads=4, dim_head=32)  # Reduced params for efficiency

        # Bidirectional Transformer Encoder (lightweight: 2 layers, 4 heads)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.out_channels, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)  # Reduced layers

        # Channel Recalibration Module (CRM) - copied/adapted from original, lightweight reduction
        self.inplanes = self.out_channels
        self.crm = self._make_layer(SEBasicBlock, self.out_channels, 2, kernel_size=3)  # Reduced blocks from 3 to 2

        # Classification Head
        # feature_dim debe coincidir con out_channels después de CRM (out_channels * expansion = out_channels)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # Calcular feature_dim dinámicamente: out_channels después de CRM
        feature_dim = self.out_channels * SEBasicBlock.expansion
        self.clf = nn.Linear(feature_dim, configs.num_classes)

        # Morphology Loss Auxiliary (simple MSE on attention focus, e.g., QRS region mask)
        self.morph_loss_weight = 0.1  # Hyperparam
        # Assume a fixed QRS mask for simplicity (in practice, compute based on R-peak)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x_in, use_attn=False, compute_morph_loss=False):
        # x_in: b, c, l (típicamente b, 1, l para ECG de un canal)
        batch_size, input_channels, seq_len = x_in.shape
        
        # Denoising (applied to input signal)
        x = self.denoise(x_in)  # b, mid_channels, l

        # Time-Frequency Transform (STFT to 2D spectrogram)
        # STFT funciona mejor con señal 1D, procesamos cada canal por separado si hay múltiples
        # Para ECG típicamente es 1 canal
        if x.shape[1] > 1:
            # Si hay múltiples canales, tomar el primero o promediar
            x_stft = x[:, 0, :]
        else:
            x_stft = x.squeeze(1)
        
        # STFT: compatible con diferentes versiones de PyTorch
        stft_complex = torch.stft(x_stft, n_fft=self.stft_n_fft, hop_length=self.stft_hop_length,
                                  win_length=self.stft_win_length, return_complex=True,
                                  normalized=False, onesided=True, pad_mode='reflect')
        # Convertir a magnitud
        spec = torch.abs(stft_complex).unsqueeze(1)  # b, 1, f, t (magnitude spectrogram)

        # TF-MG-MSC
        x1 = self.relu(self.conv2d_1(spec))
        x2 = self.relu(self.conv2d_2(spec))
        x3 = self.relu(self.conv2d_3(spec))

        # DSF with Gated Attention
        x_cat = torch.cat([x1, x2, x3], dim=1)  # b, 48, f, t (16*3=48)
        x_cat = self.bn2d(x_cat)
        weights = self.gate(x_cat)  # b,3,1,1
        x_fused = weights[:, 0].unsqueeze(1) * x1 + weights[:, 1].unsqueeze(1) * x2 + weights[:, 2].unsqueeze(1) * x3
        x_fused = self.dropout(x_fused)

        # Rearrange to sequence for VT-SA (patch-like: flatten freq-time to seq, channels as dim)
        b, c, f, t = x_fused.shape
        x_seq = rearrange(x_fused, 'b c f t -> b (f t) c')  # b, seq_len, dim (seq_len=f*t, dim=16)

        # VT-SA
        x_vt, vt_weights = self.vt_sa(x_seq, use_attn=use_attn)

        # Bidirectional Transformer (forward + reverse)
        x_fwd = self.transformer_encoder(x_vt)
        x_rev = self.transformer_encoder(torch.flip(x_vt, [1]))
        x_trans = x_fwd + x_rev  # b, seq, dim

        # Reshape back to 2D (freq, time) y luego promediar sobre dimensiones frecuenciales
        # para obtener representación temporal 1D
        x_trans = rearrange(x_trans, 'b (f t) c -> b c f t', f=f, t=t)  # b, c, f, t
        
        # Promediar sobre dimensión de frecuencia para obtener representación temporal
        # Alternativa: usar AdaptiveAvgPool2d para reducir a longitud original
        x_trans_2d = x_trans.mean(dim=2)  # b, c, t (promedio sobre frecuencia)
        
        # Interpolar/redimensionar a longitud original si es necesario
        if x_trans_2d.shape[2] != seq_len:
            x_trans_2d = F.interpolate(x_trans_2d, size=seq_len, mode='linear', align_corners=False)

        # CRM (espera input b, c, l)
        x = self.crm(x_trans_2d)

        # Head
        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.clf(x_flat)

        # Morphology Loss (auxiliary: simple example, focus on QRS region via attention weights)
        # Assume QRS is central part of signal; in practice, use R-peak detection
        if compute_morph_loss and use_attn and vt_weights is not None:
            # Example: encourage attention on central seq (QRS approx)
            tf_seq_len = x_seq.size(1)  # Longitud de la secuencia tiempo-frecuencia
            qrs_mask = torch.zeros(tf_seq_len, device=x_in.device)
            # Asumir que QRS está en la región central (ajustar según necesidad)
            qrs_start = tf_seq_len // 4
            qrs_end = 3 * tf_seq_len // 4
            qrs_mask[qrs_start:qrs_end] = 1.0  # Máscara para región QRS aproximada
            # Normalizar máscara
            qrs_mask = qrs_mask / (qrs_mask.sum() + 1e-8)
            
            # Calcular distribución de atención promedio sobre queries y heads
            attn_focus = vt_weights.mean(dim=1).mean(dim=1)  # b, seq_len (promedio sobre heads y queries)
            # Normalizar atención
            attn_focus = attn_focus / (attn_focus.sum(dim=1, keepdim=True) + 1e-8)
            
            # Calcular pérdida como KL divergence o MSE
            qrs_mask_expanded = qrs_mask.unsqueeze(0).repeat(batch_size, 1)
            morph_loss = F.mse_loss(attn_focus, qrs_mask_expanded) * self.morph_loss_weight
        else:
            morph_loss = None

        return logits, morph_loss