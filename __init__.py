import torch


def randn_like(cond, generator=None):
    return torch.randn(cond.size(), generator=generator).to(cond)


# From samplers.py
COND = 0
UNCOND = 1


class CADS:
    current_step = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_scale": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.25}),
                "t1": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6}),
                "t2": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.9}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "optional": {
                "rescale": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0}),
                "start_step": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "total_steps": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "apply_to": (["uncond", "cond", "both"],),
                "key": (["y", "c_crossattn"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "do"

    CATEGORY = "utils"

    def do(self, model, noise_scale, t1, t2, seed, rescale=0.0, start_step=-1, total_steps=-1, apply_to="both", key="y"):
        previous_wrapper = model.model_options.get("model_function_wrapper")

        im = model.model.model_sampling
        CADS.current_step = start_step
        generator = torch.Generator(device="cpu").manual_seed(seed)

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
                t = 1.0 - min(1.0, max(CADS.current_step / total_steps, 0.0))
                CADS.current_step += 1

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
            s = noise_scale
            noise = randn_like(y, generator=generator)
            gamma = torch.tensor(gamma).to(y)
            psi = rescale
            if psi > 0:
                y_mean, y_std = y.mean(), y.std()
            y = gamma.sqrt().item() * y + s * (1 - gamma).sqrt().item() * noise
            # FIXME: does this work at all like it's supposed to?
            if psi > 0:
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

            if noise_scale > 0.0:
                gamma = cads_gamma(timestep)
                for i in range(c[key].size(dim=0)):
                    if cond_or_uncond[i % len(cond_or_uncond)] == skip:
                        continue
                    c[key][i] = cads_noise(gamma, c[key][i])

            if previous_wrapper:
                return previous_wrapper(apply_model, args)

            return apply_model(input_x, timestep, **c)

        m = model.clone()
        m.set_model_unet_function_wrapper(apply_cads)

        return (m,)


NODE_CLASS_MAPPINGS = {"CADS": CADS}
