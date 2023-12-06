# CADS implementation for ComfyUI

Attempts to implement [CADS](https://arxiv.org/abs/2310.17347) for ComfyUI.

Credit also to the [A1111 implementation](https://github.com/v0xie/sd-webui-cads/tree/main) that I used as a reference.

# Usage

Apply the node to a model and set `noise_scale` > 0.0.

The node sets a unet wrapper function, but attempts to preserve any existing wrappers, so apply it after other nodes that set a unet wrapper function, and it might still work.

`t1` and `t2` affect the scaling of the added noise; after `t1`, the noise scales down until `t2`, after which no noise is added anymore and the unnoised prompt is used

`start_step` and `total_steps` are optional values that affect how the noise scaling schedule is calculated. If `start_step` is greater or equal to `total_steps`, the algorithm uses the sampler's timestep value instead which is not necessarily linear as it's affected by the sampler scheduler.

The `rescale` parameter applies optional normalization to the noised conditioning. It's disabled at 0.

# Bugs


The implementation might not be correct at all; I'm not 100% clear on the math as to where the noise is actually supposed to be added.
and I couldn't make it produce quite the same results as the A1111 node. The algorithm still seems to help with variety though.

Not tested with SDXL. Might do weird things.

I'm not sure if the rescale parameter does anything useful, but feel free to experiment.
