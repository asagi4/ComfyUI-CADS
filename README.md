# Experimental CADS implementation for ComfyUI

Attempts to implement [CADS](https://arxiv.org/abs/2310.17347) for ComfyUI.

Credit also to the [A1111 implementation](https://github.com/v0xie/sd-webui-cads/tree/main) that I used as a reference.

There isn't any real way to tell what effect CADS will have on your generations, but you can load [this example workflow](example_workflows/CADScompare.json?raw=1) into ComfyUI to compare between CADS and non-CADS generations.

# Usage

Apply the node to a model and set `noise_scale` to a nonzero value. The scale can also be negative, but values too far from 0 will result in garbage unless `rescale` is also used.

The `rescale` parameter applies normalization to the noised conditioning and combines them with a weighted sum. It's disabled at 0 and at 1, only the normalized value is used.

The node sets a unet wrapper function, but attempts to preserve any existing wrappers, so apply it after other nodes that set a unet wrapper function, and it might still work.

`t1` and `t2` affect the scaling of the added noise; after `t2`, the noise scales down until `t1`, after which no noise is added anymore and the unnoised prompt is used. The diffusion process runs **backwards** from 1 to 0, so `t2` is greater than `t1`.

`start_step` and `total_steps` are optional values that affect how the noise scaling schedule is calculated. If `start_step` is greater or equal to `total_steps`, the algorithm uses the sampler's timestep value instead which is not necessarily linear as it's affected by the sampler scheduler.


`apply_to` allows you to apply the noise selectively, defaulting to `uncond`. `key` selects where to add the noise.

`noise_type` determines the probability distribution of the generated noise.

# Bugs

Noise was previously applied to cross attention. It's now applied by default to the regular conditioning `y`, which seems to make more sense. Use the `key` parameter to restore the old behaviour.

The implementation might not be correct at all; I'm not 100% clear on the math as to where the noise is actually supposed to be added.
and I couldn't make it produce quite the same results as the A1111 node. The algorithm still seems to help with variety though.
