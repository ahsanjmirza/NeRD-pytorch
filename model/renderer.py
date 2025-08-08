import torch
import torch.nn as nn
from utils import math_utils
import numpy as np

class Renderer(nn.Module):
    def __init__(self,
        eval_background = False,
        compress_sharpness = False,
        compress_amplitude = False
    ):
        super().__init__()
        self.eval_background = eval_background
        self.compress_sharpness = compress_sharpness
        self.compress_amplitude = compress_amplitude
        return
                    
    def forward(self, 
        sg_illumination,
        basecolor,
        metallic,
        roughness,
        normal,
        alpha,
        view_dirs

    ):

        lin_basecolor = math_utils.srgb_to_linear(basecolor)
        diffuse = lin_basecolor * (1. - metallic) 
        specular = math_utils.mix(
            torch.ones_like(lin_basecolor) * 0.04, lin_basecolor, metallic
        )

        normal = torch.where(normal == torch.zeros_like(normal), view_dirs, normal)

        specular = specular[:, None, ...]
        diffuse = diffuse[:, None, ...]
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
            if len(alpha.shape) == 1:
                alpha = alpha[:, None]
            alpha = torch.clip(alpha, 0.0, 1.0)
            return torch.nn.functional.relu(brdf * alpha + env * (1 - alpha))
        else:
            return torch.nn.functional.relu(brdf)
    
    def __extract_sg_components(self, sg):
        s_amplitude = (
            math_utils.safe_exp(sg[..., 0:3])
            if self.compress_amplitude
            else sg[..., 0:3]
        )
        s_axis = sg[..., 3:6]
        s_sharpness = (
            math_utils.safe_exp(sg[..., 6:7])
            if self.compress_sharpness
            else sg[..., 6:7]
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
    
    def __stack_sg_components(self, s_amplitude, s_axis, s_sharpness):
        return torch.concat(
            [
                math_utils.safe_log(s_amplitude)
                if self.compress_amplitude
                else s_amplitude,
                s_axis,
                math_utils.safe_log(math_utils.saturate(s_sharpness, 0.5, 30))
                if self.compress_sharpness
                else s_sharpness,
            ],
            dim = -1
        )
    
    def __distribution_term(self, d, roughness):
        a2 = math_utils.saturate(roughness * roughness, 1e-3)
        tmp = np.pi * a2
        ret = self.__stack_sg_components(
            math_utils.to_vec3(math_utils.recp_no_nan(tmp)),
            d,
            2.0 / torch.maximum(a2, torch.ones_like(a2) * 1e-6)
        )
        return ret
    
    def __sg_wrap_distribution(self, ndf, v):
        ndf_amplitude, ndf_axis, ndf_sharpness = self.__extract_sg_components(ndf)
        ret = torch.concat(
            [
                ndf_amplitude,
                math_utils.reflect(-v, ndf_axis),
                math_utils.div_no_nan(
                    ndf_sharpness,
                    (4.0 * math_utils.saturate(math_utils.dot(ndf_axis, v), 1e-4))
                )
            ],
            dim = -1
        )
        return ret
    
    def __sg_integral(self, sg):
        s_amplitude, _, s_sharpness = self.__extract_sg_components(sg)
        expTerm = 1.0 - math_utils.safe_exp(-2.0 * s_sharpness)
        ret = 2 * np.pi * math_utils.div_no_nan(s_amplitude, s_sharpness) * expTerm
        return ret
    
    def __sg_inner_product(self, sg1, sg2):
        s1_amplitude, s1_axis, s1_sharpness = self.__extract_sg_components(sg1)
        s2_amplitude, s2_axis, s2_sharpness = self.__extract_sg_components(sg2)
        umLength = math_utils.magnitude(
            s1_sharpness * s1_axis + s2_sharpness * s2_axis
        )[..., None]
        expo = (
            math_utils.safe_exp(umLength - s1_sharpness - s2_sharpness)
            * s1_amplitude
            * s2_amplitude
        )
        other = 1.0 - math_utils.safe_exp(-2.0 * umLength)
        return math_utils.div_no_nan(2.0 * np.pi * expo * other, umLength)
    
    def __evaluate_diffuse(self, sg_illumination, diffuse, normal):
        diff = diffuse / np.pi

        _, s_axis, s_sharpness = self.__extract_sg_components(sg_illumination)
        mudn = math_utils.saturate(math_utils.dot(s_axis, normal))

        c0 = 0.36
        c1 = 1.0 / (4.0 * c0)

        eml = math_utils.safe_exp(-s_sharpness)
        em2l = eml * eml
        rl = math_utils.recp_no_nan(s_sharpness)

        scale = 1.0 + 2.0 * em2l - rl
        bias = (eml - em2l) * rl - em2l

        x = math_utils.safe_sqrt(1.0 - scale)
        x0 = c0 * mudn
        x1 = c1 * x

        n = x0 + x1

        y_cond = torch.le(torch.abs(x0), x1)
        y_true = n * (n / torch.maximum(x, torch.ones_like(x) * 1e-6))
        y_false = mudn
        y = torch.where(y_cond, y_true, y_false)

        res = scale * y + bias
        res = res * self.__sg_integral(sg_illumination) * diff

        return res
    
    def __ggx(self, a2, ndx):
        return math_utils.recp_no_nan(ndx + math_utils.safe_sqrt(a2 + (1 - a2) * ndx * ndx))
    
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
        G = self.__ggx(a2, ndl) * self.__ggx(a2, ndv)
        powTerm = torch.pow(1.0 - ldh, 5)
        F = specular + (1.0 - specular) * powTerm

        output = D * G * F * ndl
        return torch.nn.functional.relu(output)
    
    def __brdf_eval(self,
        sg_illumination,
        diffuse,
        specular,
        roughness,
        normal,
        view_dirs
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