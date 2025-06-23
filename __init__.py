import torch
from math import sqrt


def generate_noise(cond, generator=None, noise_type="normal"):
    t = torch.empty_like(cond)
    if noise_type == "uniform":
        return t.uniform_()
    elif noise_type == "exponential":
        return t.exponential_()
    else:
        return t.normal_()


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
                "noise_scale": ("FLOAT", {"min": -5.0, "max": 5.0, "step": 0.01, "default": 0.25}),
                "t1": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6}),
                "t2": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.9}),
            },
            "optional": {
                "rescale": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0}),
                "start_step": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "total_steps": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "apply_to": (["uncond", "cond", "both"],),
                "key": (["both", "y", "c_crossattn"],),
                "noise_type": (["normal", "uniform", "exponential"],),
                "cfg_mode": (["no", "yes"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "do"

    CATEGORY = "utils"

    def do(
        self,
        model,
        noise_scale,
        t1,
        t2,
        rescale=0.0,
        start_step=-1,
        total_steps=-1,
        apply_to="both",
        key="y",
        noise_type="normal",
        cfg_mode="no",
    ):
        previous_wrapper = model.model_options.get("model_function_wrapper")

        if key == "both":
            keys = ["y", "c_crossattn"]
        else:
            keys = [key]

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

        def cads_noise(gamma, y):
            if y is None:
                return None
            noise = generate_noise(y, noise_type=noise_type)
            psi = rescale
            if psi != 0:
                y_mean, y_std = y.mean(), y.std()
            y = sqrt(gamma) * y + noise_scale * sqrt(1 - gamma) * noise
            # FIXME: does this work at all like it's supposed to?
            if psi != 0:
                y_scaled = (y - y.mean()) / y.std() * y_std + y_mean
                if not y_scaled.isnan().any():
                    y = psi * y_scaled + (1 - psi) * y
                else:
                    print("Warning, NaNs during rescale")
            return y

        def apply_cads(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            cond_or_uncond = args["cond_or_uncond"]
            c = args["c"]

            if noise_scale != 0.0:
                for key in keys:
                    if key not in c:
                        continue
                    noise_target = c[key].clone()
                    gamma = cads_gamma(timestep)
                    for i in range(noise_target.size(dim=0)):
                        if cond_or_uncond[i % len(cond_or_uncond)] == skip:
                            continue
                        noise_target[i] = cads_noise(gamma, noise_target[i])
                    c[key] = noise_target

            if previous_wrapper:
                return previous_wrapper(apply_model, args)

            return apply_model(input_x, timestep, **c)

        def pre_cfg_func(args):
            cond, uncond = args["conds_out"]
            timestep = args["timestep"]
            if noise_scale != 0.0:
                gamma = cads_gamma(timestep)
                if apply_to in ["cond", "both"]:
                    cond = cads_noise(gamma, cond)
                if uncond is not None and apply_to in ["uncond", "both"]:
                    uncond = cads_noise(gamma, uncond)
            return [cond, uncond]


        m = model.clone()
        if cfg_mode == "no":
            m.set_model_unet_function_wrapper(apply_cads)
        else:
            m.set_model_sampler_pre_cfg_function(pre_cfg_func)

        return (m,)


NODE_CLASS_MAPPINGS = {"CADS": CADS}
