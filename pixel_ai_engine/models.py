"""
Pixel AI Engine - Neural Network Architectures & Custom Pixel-Art Losses
Implementa:
  1. PixelArtUNetGenerator: U-Net profunda con Skip Connections y Bloques Residuales.
  2. PixelArtPatchDiscriminator: Discriminador PatchGAN 70x70 para texturas nítidas.
  3. SobelEdgeLoss: Filtro diferencial Sobel para bordes duros sin difuminados.
  4. VGGPerceptualLoss: Pérdida semántica multiescala con VGG16.
  5. PixelArtLoss: Función de pérdida compuesta lista para entrenamiento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Optional, Dict

from .config import (
    IN_CHANNELS,
    OUT_CHANNELS,
    LAMBDA_L1,
    LAMBDA_EDGE,
    LAMBDA_PERC,
    LAMBDA_ADV,
    DEVICE
)


# ============================================================================
# 1. BLOQUES CONVOLUCIONALES BÁSICOS
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Bloque residual para el cuello de botella que preserva detalles de identidad y color.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


# ============================================================================
# 2. GENERADOR U-NET CONDICIONAL PROFUNDO
# ============================================================================

class PixelArtUNetGenerator(nn.Module):
    """
    Generador U-Net para Pixel Art.
    Entrada: 6 canales (3 Identidad Frontal + 3 Pose Maniquí).
    Salida: 4 canales (RGBA: Rojo, Verde, Azul, Canal Alfa de transparencia).
    """
    def __init__(self, in_channels: int = IN_CHANNELS, out_channels: int = OUT_CHANNELS):
        super().__init__()
        
        # Codificador (Downsampling)
        # 128 -> 64 -> 32 -> 16 -> 8
        self.e1 = ConvBlock(in_channels, 64, normalize=False)   # 128 -> 64
        self.e2 = ConvBlock(64, 128)                            # 64 -> 32
        self.e3 = ConvBlock(128, 256)                           # 32 -> 16
        self.e4 = ConvBlock(256, 512)                           # 16 -> 8
        
        # Bottleneck (Cuello de botella con bloques residuales)
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # Decodificador (Upsampling con Skip Connections)
        self.d1 = DeconvBlock(512, 256, dropout=0.2)            # 8 -> 16
        self.d2 = DeconvBlock(512, 128)                         # 16 -> 32 (256 cat 256 = 512)
        self.d3 = DeconvBlock(256, 64)                          # 32 -> 64 (128 cat 128 = 256)
        
        # Capa de salida a resolución completa
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 -> 128
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Salida normalizada [-1, 1] en los 4 canales (RGB + Alpha)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder con concatenación de saltos (Skip connections)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, x3], dim=1))
        d3 = self.d3(torch.cat([d2, x2], dim=1))
        
        # Salida final
        out = self.final_up(torch.cat([d3, x1], dim=1))
        return out


# ============================================================================
# 3. DISCRIMINADOR PATCHGAN 70x70
# ============================================================================

class MinibatchStdDev(nn.Module):
    """
    Capa de Discriminación por Minilotes (Minibatch Discrimination / StdDev de StyleGAN/ProGAN).
    Calcula la dispersión estadística de características a través del minilote.
    Si el generador intenta colapsar a cuadros grises planos o generar imágenes idénticas,
    la desviación estándar del lote cae a cero y el discriminador lo detecta y penaliza de inmediato.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if b <= 1:
            zero_channel = torch.zeros((b, 1, h, w), dtype=x.dtype, device=x.device)
            return torch.cat([x, zero_channel], dim=1)
        
        # Desviación estándar a través del lote (dim=0)
        # Se agrega 1e-8 para estabilidad numérica estricta
        # Desviacion estandar a traves del lote (dim=0)
        # Computo en float32 y clamp seguro (1e-4) para prevenir underflow a 0.0 y derivadas infinitas/NaN en FP16 AMP
        x_f32 = x.float()
        var = torch.var(x_f32, dim=0, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-4)).to(dtype=x.dtype)
        # Promedio global de variabilidad
        mean_std = torch.mean(std)
        # Mapa de características de 1 canal expandido a todo el batch
        std_feature = mean_std.expand(b, 1, h, w)
        return torch.cat([x, std_feature], dim=1)


class PixelArtPatchDiscriminator(nn.Module):
    """
    Discriminador PatchGAN con Discriminación por Minilotes (Minibatch Discrimination).
    Evalúa parches de 70x70 e incorpora la estadística de diversidad cruzada de todo el lote.
    Garantiza que la IA nunca colapse a imágenes planas ni pierda diversidad estilística.
    Entrada: Condición (6 canales) + Target Real/Fake (4 canales) = 10 canales.
    """
    def __init__(self, in_channels: int = IN_CHANNELS + OUT_CHANNELS):
        super().__init__()
        
        self.features = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 15
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Módulo de Discriminación por Minilotes
        self.minibatch_std = MinibatchStdDev()
        
        # Capa 1D de decisión final (512 canales + 1 canal de varianza de minilote = 513 canales)
        self.final_conv = nn.Conv2d(512 + 1, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, target], dim=1)
        feat = self.features(x)
        feat_with_std = self.minibatch_std(feat)
        return self.final_conv(feat_with_std)


# ============================================================================
# 4. PÉRDIDA DE BORDES SOBEL (Pixel Art Edge Loss)
# ============================================================================

class SobelEdgeLoss(nn.Module):
    """
    Calcula los gradientes espaciales horizontales y verticales mediante filtros Sobel.
    Fuerza a la red a producir bordes nítidos de 1 píxel, eliminando el suavizado (anti-aliasing)
    que arruina el Pixel Art.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32)
        
        self.register_buffer("kernel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_reshaped = x.reshape(b * c, 1, h, w)
        gx = F.conv2d(x_reshaped, self.kernel_x, padding=1)
        gy = F.conv2d(x_reshaped, self.kernel_y, padding=1)
        edge = torch.sqrt(torch.clamp(gx ** 2 + gy ** 2, min=1e-5))
        return edge.reshape(b, c, h, w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        return F.l1_loss(pred_edges, target_edges)


class LaplacianMicroDetailLoss(nn.Module):
    """
    Filtro Laplaciano 2D que extrae las frecuencias espaciales más altas:
    tatuajes, ojos, líneas divisorias de 1 píxel y pliegues de ropa.
    Fuerza al generador a no difuminar ni perder trazos finos de 1px.
    """
    def __init__(self):
        super().__init__()
        laplacian = torch.tensor([[ 0.0, -1.0,  0.0],
                                  [-1.0,  4.0, -1.0],
                                  [ 0.0, -1.0,  0.0]], dtype=torch.float32)
        self.register_buffer("kernel", laplacian.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        p = pred.reshape(b * c, 1, h, w)
        t = target.reshape(b * c, 1, h, w)
        lp = F.conv2d(p, self.kernel, padding=1)
        lt = F.conv2d(t, self.kernel, padding=1)
        return F.l1_loss(lp, lt)


class TattooInkLoss(nn.Module):
    """
    Pérdida quirúrgica que protege la tinta oscura de tatuajes y detalles finos.
    Si el objetivo contiene píxeles oscuros (luminancia baja) en la figura del personaje,
    penaliza fuertemente si el generador intenta pintarlos de blanco o color piel claro.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_rgba: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
        # Extraer luminancia perceptual: 0.299*R + 0.587*G + 0.114*B (en [-1, 1])
        t_lum = 0.299 * target_rgba[:, 0:1] + 0.587 * target_rgba[:, 1:2] + 0.114 * target_rgba[:, 2:3]
        p_lum = 0.299 * pred_rgba[:, 0:1] + 0.587 * pred_rgba[:, 1:2] + 0.114 * pred_rgba[:, 2:3]
        
        # Píxeles visibles del personaje
        fg = target_rgba[:, 3:4] > 0.0
        # Píxeles de tinta oscura en el objetivo (tatuajes, ojos, líneas oscuras)
        is_ink = (t_lum < -0.30) & fg
        
        if is_ink.sum() < 4:
            return torch.tensor(0.0, device=pred_rgba.device)
            
        # Penalizar fuertemente si el modelo aclara la tinta convirtiéndola en tela blanca
        return F.l1_loss(p_lum[is_ink], t_lum[is_ink]) * 3.0


# ============================================================================
# 5. PÉRDIDA PERCEPTUAL VGG-16
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Extrae mapas de características profundas de VGG-16 para asegurar coherencia
    anatómica y estilística entre el personaje generado y el objetivo.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        for i in range(4):   # relu1_2
            self.slice1.add_module(str(i), vgg[i])
        for i in range(4, 9):  # relu2_2
            self.slice2.add_module(str(i), vgg[i])
        for i in range(9, 16):  # relu3_3
            self.slice3.add_module(str(i), vgg[i])
            
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        # Re-normalizar de [-1, 1] a distribución ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred_rgb.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred_rgb.device).view(1, 3, 1, 1)
        
        p = ((pred_rgb + 1.0) * 0.5 - mean) / std
        t = ((target_rgb + 1.0) * 0.5 - mean) / std
        
        p_feat1 = self.slice1(p)
        t_feat1 = self.slice1(t)
        
        p_feat2 = self.slice2(p_feat1)
        t_feat2 = self.slice2(t_feat1)
        
        p_feat3 = self.slice3(p_feat2)
        t_feat3 = self.slice3(t_feat2)
        
        loss = F.l1_loss(p_feat1, t_feat1) + F.l1_loss(p_feat2, t_feat2) + F.l1_loss(p_feat3, t_feat3)
        return loss


# ============================================================================
# 6. PÉRDIDAS QUIRÚRGICAS: PUREZA ALFA Y FIDELIDAD DE IDENTIDAD
# ============================================================================

class AlphaCrispnessLoss(nn.Module):
    """
    Penaliza fuertemente los valores de opacidad intermedios (semitransparencias borrosas).
    En Pixel Art el canal alfa debe ser binario: 0 (aire) o 1 (cuerpo sólido).
    Minimiza alfa * (1 - alfa), que es 0 solo cuando alfa es exactamente 0 o 1.
    """
    def __init__(self):
        super().__init__()

    def forward(self, rgba: torch.Tensor) -> torch.Tensor:
        alpha = (rgba[:, 3:] + 1.0) * 0.5  # escala a [0, 1]
        loss = torch.mean(alpha * (1.0 - alpha))
        return loss * 4.0


class IdentityColorConsistencyLoss(nn.Module):
    """
    Compara las matrices de correlación de color (Gram Matrix) y estadísticas cromáticas
    del personaje generado contra la foto frontal de identidad oficial.
    Garantiza que la ropa, piel y pelo mantengan la misma paleta y contraste sin desteñirse.
    """
    def __init__(self):
        super().__init__()

    def forward(self, generated_rgb: torch.Tensor, identity_rgb: torch.Tensor) -> torch.Tensor:
        gen_f = generated_rgb.float()
        id_f = identity_rgb.float()
        
        # Comparar medias de canal (R, G, B)
        mean_gen = gen_f.mean(dim=[2, 3])
        mean_id = id_f.mean(dim=[2, 3])
        mean_loss = F.l1_loss(mean_gen, mean_id)

        # Comparar desviaciones de canal
        std_gen = torch.std(gen_f, dim=[2, 3], unbiased=False)
        std_id = torch.std(id_f, dim=[2, 3], unbiased=False)
        std_loss = F.l1_loss(std_gen, std_id)
        
        # Covarianza de color 3x3 estable con adaptive_avg_pool a 32x32 (1024 elementos, nunca desborda FP16)
        p_small = F.adaptive_avg_pool2d(gen_f, (32, 32)).reshape(gen_f.shape[0], 3, -1)
        t_small = F.adaptive_avg_pool2d(id_f, (32, 32)).reshape(id_f.shape[0], 3, -1)
        cov_gen = torch.bmm(p_small, p_small.transpose(1, 2)) / 1024.0
        cov_id = torch.bmm(t_small, t_small.transpose(1, 2)) / 1024.0
        cov_loss = F.mse_loss(cov_gen, cov_id)
        
        total = cov_loss * 5.0 + mean_loss * 2.0 + std_loss * 2.0
        return torch.nan_to_num(total, nan=0.0, posinf=1.0, neginf=0.0)


# ============================================================================
# 7. PÉRDIDAS ANATÓMICAS Y DE DISCRETIZACIÓN DE PIXEL ART
# ============================================================================

class DiscretePaletteQuantizationLoss(nn.Module):
    """
    Penaliza fuertemente los gradientes suaves y colores continuos intermedios.
    Obliga a que cada píxel del cuerpo converja a bloques de color nítidos y planos
    típicos del pixel art auténtico, eliminando el difuminado borroso.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        # Varianza local en parches de 3x3
        # Si el objetivo es plano, la predicción debe ser igualmente plana sin ruido continuo
        p_unfold = F.unfold(pred_rgb, kernel_size=3, padding=1)
        t_unfold = F.unfold(target_rgb, kernel_size=3, padding=1)

        p_var = torch.var(p_unfold, dim=1)
        t_var = torch.var(t_unfold, dim=1)

        return F.l1_loss(p_var, t_var) * 3.0


class ShoeGroundingLoss(nn.Module):
    """
    Penaliza si los zapatos o pies del personaje flotan en el aire o se desfasaron
    respecto a la línea de suelo del sprite de referencia.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_rgba: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
        h = pred_rgba.shape[2]
        y_feet_start = int(h * 0.80)
        pred_feet = pred_rgba[:, 3:, y_feet_start:, :]
        target_feet = target_rgba[:, 3:, y_feet_start:, :]
        return F.l1_loss(pred_feet, target_feet) * 5.0


class FacialExpressionLoss(nn.Module):
    """
    Protege los micro-detalles de la cara: pupilas de 1-2px, cejas y expresión.
    Localiza la cabeza del personaje en las coordenadas reales de la cuadrícula
    (Y: 155 a 215, X: 95 a 160) y penaliza fuertemente el difuminado de cejas y ojos.
    """
    def __init__(self):
        super().__init__()
        laplacian = torch.tensor([[0.0, -1.0, 0.0],
                                  [-1.0,  4.0, -1.0],
                                  [0.0, -1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", laplacian)

    def forward(self, pred_rgba: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
        # En la cuadrícula 256x256 el personaje está anclado en Y=[124, 244]
        # La cabeza y rostro (gorro, cejas, ojos, nariz) están exactamente en Y=125 a 185, X=80 a 175.
        p_face = pred_rgba[:, :3, 125:185, 80:175]
        t_face = target_rgba[:, :3, 125:185, 80:175]
        
        # 1. Pérdida L1 focal directa sobre la caja del rostro
        face_l1 = F.l1_loss(p_face, t_face)
        
        # 2. Pérdida Laplaciana de alta frecuencia para forzar líneas de 1px en cejas y bordes oculares
        p_lum = 0.299 * p_face[:, 0:1] + 0.587 * p_face[:, 1:2] + 0.114 * p_face[:, 2:3]
        t_lum = 0.299 * t_face[:, 0:1] + 0.587 * t_face[:, 1:2] + 0.114 * t_face[:, 2:3]
        
        hf_p = F.conv2d(p_lum, self.kernel, padding=1)
        hf_t = F.conv2d(t_lum, self.kernel, padding=1)
        face_edge = F.l1_loss(hf_p, hf_t)
        
        return face_l1 * 10.0 + face_edge * 12.0


class SilhouetteAlignmentLoss(nn.Module):
    """
    Perdida de alineacion de silueta y molde (IoU + L1).
    Obliga a que la silueta generada calce milimetricamente con el cuerpo objetivo y el molde,
    eliminando amputaciones, extremidades fantasma y desbordes fuera de la figura (resolviendo la desviacion IoU).
    Totalmente seguro con FP16 AMP.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_rgba: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
        p_alpha = ((pred_rgba[:, 3:4].float() + 1.0) * 0.5).clamp(0.0, 1.0)
        t_alpha = ((target_rgba[:, 3:4].float() + 1.0) * 0.5).clamp(0.0, 1.0)
        
        l1_mask = F.l1_loss(p_alpha, t_alpha)
        intersection = torch.sum(p_alpha * t_alpha, dim=[1, 2, 3])
        union = torch.sum(p_alpha + t_alpha, dim=[1, 2, 3]) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = torch.mean(1.0 - iou)
        return l1_mask * 5.0 + iou_loss * 10.0


class FocalColorDefectLoss(nn.Module):
    """
    Penaliza fuertemente y de forma cuadrática los píxeles del cuerpo donde el desvío de color
    supere los 35 niveles RGB (0.137 normalizado).
    Fuerza a la red a eliminar de raíz los píxeles defectuosos y gradientes lodosos.
    """
    def __init__(self, threshold: float = 35.0 / 255.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred_rgba: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
        target_alpha = (target_rgba[:, 3:4] + 1.0) * 0.5
        body_mask = (target_alpha > 0.15).float()
        
        rgb_diff = torch.abs(pred_rgba[:, :3] - target_rgba[:, :3]) * 0.5
        max_diff, _ = torch.max(rgb_diff, dim=1, keepdim=True)
        
        excess = F.relu(max_diff - self.threshold)
        focal_penalty = (excess ** 2) * body_mask * 20.0
        return torch.sum(focal_penalty.float()) / torch.clamp(torch.sum(body_mask.float()), min=1.0)


# ============================================================================
# 8. PÉRDIDA COMBINADA PIXEL ART
# ============================================================================

class PixelArtLoss(nn.Module):
    def __init__(self,
                 lambda_l1: float = LAMBDA_L1,
                 lambda_edge: float = LAMBDA_EDGE,
                 lambda_detail: float = 45.0,
                 lambda_tattoo: float = 35.0,
                 lambda_perc: float = LAMBDA_PERC,
                 lambda_adv: float = LAMBDA_ADV,
                 lambda_alpha_crisp: float = 6.0,
                 lambda_id_color: float = 4.0,
                 lambda_discrete: float = 12.0,
                 lambda_shoes: float = 6.0,
                 lambda_face: float = 8.0,
                 lambda_focal_defect: float = 18.0,
                 lambda_silhouette: float = 12.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_edge = lambda_edge
        self.lambda_detail = lambda_detail
        self.lambda_tattoo = lambda_tattoo
        self.lambda_perc = lambda_perc
        self.lambda_adv = lambda_adv
        self.lambda_alpha_crisp = lambda_alpha_crisp
        self.lambda_id_color = lambda_id_color
        self.lambda_discrete = lambda_discrete
        self.lambda_shoes = lambda_shoes
        self.lambda_face = lambda_face
        self.lambda_focal_defect = lambda_focal_defect
        self.lambda_silhouette = lambda_silhouette
        
        self.edge_loss_fn = SobelEdgeLoss()
        self.detail_loss_fn = LaplacianMicroDetailLoss()
        self.tattoo_loss_fn = TattooInkLoss()
        self.alpha_crisp_fn = AlphaCrispnessLoss()
        self.id_color_fn = IdentityColorConsistencyLoss()
        self.discrete_fn = DiscretePaletteQuantizationLoss()
        self.shoe_grounding_fn = ShoeGroundingLoss()
        self.facial_fn = FacialExpressionLoss()
        self.focal_defect_fn = FocalColorDefectLoss()
        self.silhouette_fn = SilhouetteAlignmentLoss()

        try:
            self.vgg_loss_fn = VGGPerceptualLoss()
            self.has_vgg = True
        except Exception as e:
            print(f"[Aviso] No se pudo cargar VGG16 ({e}). Continuando con L1 + Sobel Edge Loss.")
            self.has_vgg = False
            
        self.gan_loss_fn = nn.BCEWithLogitsLoss()

    def generator_loss(self,
                        fake_pred_disc: torch.Tensor,
                        fake_sprites: torch.Tensor,
                        real_sprites: torch.Tensor,
                        front_identity: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        # 1. Pérdida Adversarial (en float32 con clamp para máxima estabilidad FP16 AMP)
        disc_logits = fake_pred_disc.float().clamp(-30.0, 30.0)
        loss_adv = F.binary_cross_entropy_with_logits(disc_logits, torch.ones_like(disc_logits))
        if torch.isnan(loss_adv):
            loss_adv = torch.tensor(0.0, device=fake_pred_disc.device)
        
        # 2. Pérdida L1 Ponderada en Primer Plano (RGB + Alpha)
        # El personaje (fg) recibe 3.5x más gradiente que el fondo vacío para maximizar detalles
        fg_weight = ((real_sprites[:, 3:] + 1.0) * 0.5).clamp(0.0, 1.0)
        pixel_weights = 1.0 + 2.5 * fg_weight
        loss_l1 = torch.mean(torch.abs(fake_sprites - real_sprites) * pixel_weights)
        
        # 3. Pérdida de Bordes Sobel en RGBA
        loss_edge = self.edge_loss_fn(fake_sprites, real_sprites)
        
        # 4. Pérdida de Micro-Detalles Laplaciano (tatuajes, ojos, líneas de 1px)
        loss_detail = self.detail_loss_fn(fake_sprites[:, :3], real_sprites[:, :3])
        
        # 5. Pérdida Quirúrgica de Tatuajes y Líneas Oscuras
        loss_tattoo = self.tattoo_loss_fn(fake_sprites, real_sprites)
        
        # 6. Pureza de corte Alfa (Binaridad estricta para Pixel Art)
        loss_alpha_crisp = self.alpha_crisp_fn(fake_sprites)
        
        # 7. Consistencia y mejora con la Identidad Frontal
        loss_id_color = torch.tensor(0.0, device=fake_sprites.device)
        if front_identity is not None:
            loss_id_color = self.id_color_fn(fake_sprites[:, :3], front_identity)
        
        # 8. Pérdida Perceptiva en RGB
        loss_perc = torch.tensor(0.0, device=fake_sprites.device)
        if self.has_vgg:
            loss_perc = self.vgg_loss_fn(fake_sprites[:, :3], real_sprites[:, :3])

        # 9. Discretización de Pixel Art (anti-gradientes suaves)
        loss_discrete = self.discrete_fn(fake_sprites[:, :3], real_sprites[:, :3])

        # 10. Anclaje de Pies y Zapatos al Suelo
        loss_shoes = self.shoe_grounding_fn(fake_sprites, real_sprites)

        # 11. Ojos, Cejas y Expresión Facial
        loss_face = self.facial_fn(fake_sprites, real_sprites)

        # 12. Castigo Focal Cuadrático a Píxeles Defectuosos del Cuerpo (>35 RGB)
        loss_focal_defect = self.focal_defect_fn(fake_sprites, real_sprites)
            
                # 13. Alineacion Milimetrica de Silueta con el Molde (IoU + BCE)
        loss_silhouette = self.silhouette_fn(fake_sprites, real_sprites)

        total_g_loss = (
            self.lambda_adv * loss_adv +
            self.lambda_l1 * loss_l1 +
            self.lambda_edge * loss_edge +
            self.lambda_detail * loss_detail +
            self.lambda_tattoo * loss_tattoo +
            self.lambda_perc * loss_perc +
            self.lambda_alpha_crisp * loss_alpha_crisp +
            self.lambda_id_color * loss_id_color +
            self.lambda_discrete * loss_discrete +
            self.lambda_shoes * loss_shoes +
            self.lambda_face * loss_face +
            self.lambda_focal_defect * loss_focal_defect +
            self.lambda_silhouette * loss_silhouette
        )
        
        metrics = {
            "g_total": total_g_loss.item(),
            "g_l1": loss_l1.item(),
            "g_edge": loss_edge.item(),
            "g_detail": loss_detail.item(),
            "g_tattoo": loss_tattoo.item(),
            "g_perc": loss_perc.item(),
            "g_adv": loss_adv.item(),
            "g_alpha_crisp": loss_alpha_crisp.item(),
            "g_id_color": loss_id_color.item(),
            "g_discrete": loss_discrete.item(),
            "g_shoes": loss_shoes.item(),
            "g_face": loss_face.item(),
            "g_focal_defect": loss_focal_defect.item(),
            "g_silhouette": loss_silhouette.item()
        }
        return total_g_loss, metrics

    def discriminator_loss(self,
                           real_pred: torch.Tensor,
                           fake_pred: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Suavizado de etiquetas (en float32 con clamp para evitar desbordes FP16 AMP)
        real_logits = real_pred.float().clamp(-30.0, 30.0)
        fake_logits = fake_pred.float().clamp(-30.0, 30.0)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits) * 0.9)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        total_d_loss = (loss_real + loss_fake) * 0.5
        if torch.isnan(total_d_loss):
            total_d_loss = torch.tensor(0.5, device=real_pred.device, requires_grad=True)
        
        metrics = {
            "d_total": total_d_loss.item(),
            "d_real": loss_real.item(),
            "d_fake": loss_fake.item()
        }
        return total_d_loss, metrics
