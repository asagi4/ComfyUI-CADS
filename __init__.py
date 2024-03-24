import numpy as np
import torch


# Function copied from: https://github.com/v0xie/sd-webui-cads
def add_noise(y, gamma, noise_scale, psi, rescale=False):
    """ CADS adding noise to the condition

    Arguments:
    y: Input conditioning
    gamma: Noise level w.r.t t
    noise_scale (float): Noise scale
    psi (float): Rescaling factor
    rescale (bool): Rescale the condition
    """
    y_mean, y_std = torch.mean(y), torch.std(y)
    y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * torch.randn_like(y)
    if rescale:
        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
        if not torch.isnan(y_scaled).any():
            y = psi * y_scaled + (1 - psi) * y
        else:
            print("Warning: NaN encountered in rescaling")
    return y

# From samplers.py
COND = 0
UNCOND = 1


class CADS:
    current_step = 0
    last_sigma = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_scale": ("FLOAT", {"min": -100.0, "max": 100.0, "step": 0.01, "default": 0.25}),
                "t1": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6}),
                "t2": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.9}),
            },
            "optional": {
                "mixing_factor": ("FLOAT", {"min": -100.0, "max": 100.0, "step": 0.01, "default": 1.0}),
                "rescale": ("BOOLEAN", {"default": False}),
                "start_step": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "total_steps": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "apply_to": (["uncond", "cond", "both"],),
                "key": (["y", "c_crossattn"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "do"

    CATEGORY = "utils"

    def do(self, model, noise_scale, t1, t2, mixing_factor=1.0, rescale=False, start_step=-1, total_steps=-1, apply_to="both", key="y"):
        previous_wrapper = model.model_options.get("model_function_wrapper")

        im = model.model.model_sampling
        self.current_step = start_step
        self.last_sigma = None

        skip = None
        if apply_to == "cond":
            skip = UNCOND
        elif apply_to == "uncond":
            skip = COND

        def cads_gamma(sigma):
            if start_step >= total_steps:
                ts = im.timestep(sigma[0])
                t = round(ts.item() / 999.0, 2)
            else:
                sigma_max = sigma.max().item()
                if self.last_sigma is not None and sigma_max > self.last_sigma:
                    self.current_step = start_step
                t = 1.0 - min(1.0, max(self.current_step / total_steps, 0.0))
                self.current_step += 1
                self.last_sigma = sigma_max

            if t <= t1:
                r = 1.0
            elif t >= t2:
                r = 0.0
            else:
                r = (t2 - t) / (t2 - t1)
            return r

        def apply_cads(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            cond_or_uncond = args["cond_or_uncond"]
            c = args["c"]

            if noise_scale != 0.0:
                noise_target = c.get(key, c["c_crossattn"])
                gamma = cads_gamma(timestep)
                for i in range(noise_target.size(dim=0)):
                    if cond_or_uncond[i % len(cond_or_uncond)] == skip:
                        continue
                    noise_target[i] = add_noise(noise_target[i], gamma, noise_scale, mixing_factor, rescale)

            if previous_wrapper:
                return previous_wrapper(apply_model, args)

            return apply_model(input_x, timestep, **c)

        m = model.clone()
        m.set_model_unet_function_wrapper(apply_cads)

        return (m,)


NODE_CLASS_MAPPINGS = {"CADS": CADS}
