import torch
import torch.nn as nn
from utils import math_utils 
import numpy as np

class Renderer(nn.Module):
    def __init__(self,
        eval_background: bool = False,
        compress_sharpness: bool = False,
        compress_amplitude: bool = False
    ):
        super().__init__()
        self.eval_background = eval_background
        self.compress_sharpness = compress_sharpness
        self.compress_amplitude = compress_amplitude
        return
                    
    def forward(self, 
        sg_illumination: torch.Tensor,
        basecolor: torch.Tensor,
        metallic: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,
        alpha: torch.Tensor,
        view_dirs: torch.Tensor

    ):
        lin_basecolor = math_utils.srgb_to_linear(basecolor)
        diffuse = lin_basecolor * (1. - metallic) # Only diffuse is metallic is 0
        # Interpolate between 0.04 base reflectivity where non-metallic
        # and specular color (from basecolor)
        specular = math_utils.mix(
            torch.ones_like(lin_basecolor) * 0.04, lin_basecolor, metallic
        )
        normal = torch.where(normal == torch.zeros_like(normal), view_dirs, normal)

        diffuse = diffuse[:, None, ...]
        specular = specular[:, None, ...]
        roughness = roughness[:, None, ...]

        normal = math_utils.normalize(normal)[:, None, ...]
        view_dirs = math_utils.normalize(view_dirs)[:, None, ...]

        env = None
        if self.eval_background:
            # Evaluate everything for the environment
            env = self.__sg_evaluate(sg_illumination, view_dirs)
            # And sum all contributions
            env = torch.sum(env, dim = 1)

        # Evaluate BRDF
        brdf = self.__brdf_eval(
            sg_illumination, diffuse, specular, roughness, normal, view_dirs,
        )
        brdf = torch.sum(brdf, dim = 1)

        if self.eval_background:
            if len(alpha.shape == 1):
                alpha = alpha[:, None]

            alpha = torch.clip(alpha, 0.0, 1.0)

            return torch.maximum(brdf * alpha + env * (1 - alpha), 0.0)
        else:
            return torch.maximum(brdf, 0.0)
    
    def __extract_sg_components(self, sg):
        s_amplitude = (
            math_utils.safe_exp(sg[..., :3])
            if self.compress_amplitude
            else sg[..., :3]
        )
        s_axis = sg[..., 3:6]
        s_sharpness = (
            math_utils.safe_exp(sg[..., 6:])
            if self.compress_sharpness
            else sg[..., 6:]
        )
        return (
            torch.abs(s_amplitude),
            math_utils.normalize(s_axis),
            math_utils.saturate(s_sharpness, 0.5, 30)
        )

    def __sg_evaluate(self, sg, d):
        s_amplitude, s_axis, s_sharpness = self.__extract_sg_components(sg)
        cosAngle = math_utils.dot(d, s_axis)
        return s_amplitude * math_utils.safe_exp(s_sharpness * (cosAngle - 1.0))
    
    def __distribution_term(self, d, roughness):
        a2 = math_utils.saturate(roughness * roughness, 1e-3)
        tmp = np.pi * a2
        ret = self.__stack_sg_components(
            math_utils.to_vec3(torch.where(torch.eq(tmp, 0.0), 0.0, torch.reciprocal(tmp))),
            d,
            2.0 / torch.maximum(a2, 1e-6)
        )
        return ret
    
    def __sg_wrap_distribution(self, ndf, v):
        ndf_amplitude, ndf_axis, ndf_sharpness = self.__extract_sg_components(ndf)
        ret = torch.concat(
            (
                ndf_amplitude,
                math_utils.reflect(-v, ndf_axis),
                ndf_sharpness / (4.0 * math_utils.saturate(math_utils.dot(ndf_axis, v), 1e-4))
            ),
            dim = -1
        )
        return ret
    
    def __sg_integral(self, sg):
        s_amplitude, _, s_sharpness = self.__extract_sg_components(sg)

        expTerm = 1.0 - math_utils.safe_exp(-2.0 * s_sharpness)
        ret = 2 * np.pi * (s_amplitude / torch.maximum(s_sharpness, 1e-6)) * expTerm
        return ret
    
    def __sg_inner_product(self, sg1, sg2):
        s1_amplitude, s1_axis, s1_sharpness = self.__extract_sg_components(sg1)
        s2_amplitude, s2_axis, s2_sharpness = self.__extract_sg_components(sg2)

        umLength = math_utils.magnitude(
            s1_sharpness * s1_axis + s2_sharpness * s2_axis
        )

        expo = (
            math_utils.safe_exp(umLength - s1_sharpness - s2_sharpness)
            * s1_amplitude
            * s2_amplitude
        )

        other = 1.0 - math_utils.safe_exp(-2.0 * umLength)

        return (2.0 * np.pi * expo * other) / torch.maximum(umLength, 1e-6)
    
    def __evaluate_diffuse(self, sg_illumination, diffuse, normal):
        diff = diffuse / np.pi

        _, s_axis, s_sharpness = self.__extract_sg_components(sg_illumination)
        mudn = math_utils.saturate(math_utils.dot(s_axis, normal))

        c0 = 0.36
        c1 = 1.0 / (4.0 * c0)

        eml = math_utils.safe_exp(-s_sharpness)
        em2l = eml * eml
        rl = torch.where(s_sharpness == 0.0, 0.0, 1 / s_sharpness)

        scale = 1.0 + 2.0 * em2l - rl
        bias = (eml - em2l) * rl - em2l

        x = math_utils.safe_sqrt(1.0 - scale)
        x0 = c0 * mudn
        x1 = c1 * x

        n = x0 + x1

        y_cond = torch.le(torch.abs(x0), x1)
        y_true = n * (n / torch.maximum(x, 1e-6))
        y_false = mudn
        y = torch.where(y_cond, y_true, y_false)

        res = scale * y + bias

        res = res * self.__sg_integral(sg_illumination) * diff

        return res
    
    def __ggx(self, a2, ndx):
        ret = 1.0 / torch.maximum(ndx + math_utils.safe_sqrt(a2 + (1 - a2) * ndx * ndx), 1e-6)
        return ret
    
    def __evaluate_specular(self, 
        sg_illumination, 
        specular, 
        roughness, 
        warped_ndf, 
        ndl, 
        ndv, 
        ldh
    ):
        a2 = math_utils.saturate(roughness * roughness, 1e-3)
        D = self.__sg_inner_product(warped_ndf, sg_illumination)
        G = self.__ggx(a2, ndl) * self._ggx(a2, ndv)
        powTerm = torch.pow(1.0 - ldh, 5)
        F = specular + (1.0 - specular) * powTerm

        output = D * G * F * ndl
        return torch.maximum(output, 0.0)
    
    def __brdf_eval(self,
        sg_illumination: torch.Tensor,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,
        view_dirs: torch.Tensor
    ):

        ndf = self.__distribution_term(normal, roughness)

        warped_ndf = self.__sg_wrap_distribution(ndf, view_dirs)
        _, warpDir, _ = self.__extract_sg_components(warped_ndf)

        ndl = math_utils.saturate(math_utils.dot(normal, warpDir))
        ndv = math_utils.saturate(math_utils.dot(normal, view_dirs))
        h = math_utils.normalize(warpDir + view_dirs)
        ldh = math_utils.saturate(math_utils.dot(warpDir, h))

        diffuse_eval = self.__evaluate_diffuse(sg_illumination, diffuse, normal)
        specular_eval = self.__evaluate_specular(
            sg_illumination, specular, roughness, warped_ndf, ndl, ndv, ldh
        )

        return diffuse_eval + specular_eval
    
    