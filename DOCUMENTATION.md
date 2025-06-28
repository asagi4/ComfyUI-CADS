# CADS Node Documentation

This document provides detailed documentation for the CADS (Conditioning-Aware Diffusion Solvers) node in ComfyUI.

## Purpose

CADS is an experimental implementation based on the research paper [CADS: Conditioning-Aware Diffusion Solvers for Low-Latency High-Fidelity Audio Synthesis](https://arxiv.org/abs/2310.17347). While the paper focuses on audio synthesis, this ComfyUI node adapts the principles for image generation.

The core idea behind CADS is to introduce controlled noise into the conditioning signal during the diffusion process. This can help:

- **Increase Variety:** By adding randomness to the conditioning, CADS can encourage the model to explore a wider range of outputs, leading to more diverse and creative images.
- **Improve Prompt Adherence (Potentially):** In some cases, the added noise might help the model better interpret and respond to nuanced aspects of the prompt.
- **Reduce Artifacts (Potentially):** The noise injection might help mitigate certain types of artifacts or undesirable patterns in the generated images.

It's important to note that the effects of CADS can be subtle and may vary depending on the model, prompt, and other parameters. Experimentation is key to understanding how CADS can be used to achieve desired results.

This implementation also draws inspiration from the [A1111 implementation of CADS](https://github.com/v0xie/sd-webui-cads/tree/main).

## Parameters

The CADS node has several parameters that control its behavior. These are divided into "required" and "optional" inputs.

### Required Parameters

- **`model`**:
    - **Type**: `MODEL`
    - **Purpose**: This is the input model that CADS will modify. The CADS node wraps the model's function to inject noise into the conditioning.
    - **Practical Use**: Connect the output of a checkpoint loader or another model-modifying node to this input.

- **`noise_scale`**:
    - **Type**: `FLOAT`
    - **Default**: `0.25`
    - **Min/Max**: `-5.0` / `5.0`
    - **Step**: `0.01`
    - **Purpose**: This parameter controls the overall strength of the noise added to the conditioning.
    - **Practical Use**:
        - A value of `0` disables the CADS effect.
        - Positive values add noise. Small positive values (e.g., 0.1 to 0.5) can introduce subtle variations.
        - Negative values can also be used, potentially having different qualitative effects on the image.
        - Values far from 0 (e.g., > 1.0 or < -1.0) can result in overly noisy or "garbage" outputs, especially if `rescale` is not used. Start with small values and adjust incrementally.

- **`t1`**:
    - **Type**: `FLOAT`
    - **Default**: `0.6`
    - **Min/Max**: `0.0` / `1.0`
    - **Step**: `0.01`
    - **Purpose**: This parameter, along with `t2`, defines the time range during which the noise is scaled down. The diffusion process runs backwards from timestep 1.0 (maximum noise) to 0.0 (no noise). `t1` is the timestep at which the added noise becomes zero.
    - **Practical Use**:
        - `t1` should be less than or equal to `t2`.
        - If `t` (current timestep normalized between 0 and 1) is less than or equal to `t1`, no CADS noise is added (the noise scaling factor `gamma` becomes 1.0).
        - Adjusting `t1` controls how early in the sampling process the CADS noise stops being effective. A higher `t1` means noise is active for a shorter period.

- **`t2`**:
    - **Type**: `FLOAT`
    - **Default**: `0.9`
    - **Min/Max**: `0.0` / `1.0`
    - **Step**: `0.01`
    - **Purpose**: This parameter, along with `t1`, defines the time range for noise scaling. `t2` is the timestep at which the added noise starts scaling down from its full effect (where `gamma` is 0.0).
    - **Practical Use**:
        - `t2` should be greater than or equal to `t1`.
        - If `t` (current timestep normalized between 0 and 1) is greater than or equal to `t2`, the CADS noise is applied at its maximum configured strength (noise scaling factor `gamma` is 0.0).
        - Between `t1` and `t2`, the noise scales down linearly.
        - Adjusting `t2` controls when the noise starts to fade. A lower `t2` means the full noise effect is applied for a shorter duration.

### Optional Parameters

- **`rescale`** (`psi`):
    - **Type**: `FLOAT`
    - **Default**: `0.0`
    - **Min/Max**: `0.0` / `1.0`
    - **Step**: `0.01`
    - **Purpose**: This parameter (referred to as `psi` in the code comments) applies normalization to the noised conditioning and then blends it with the original noised conditioning using a weighted sum.
    - **Practical Use**:
        - When `rescale` is `0.0`, this feature is disabled, and only the direct noised conditioning is used.
        - When `rescale` is `1.0`, only the normalized and rescaled-to-original-stats version of the noised conditioning is used.
        - Values between `0.0` and `1.0` blend the two: `rescale * normalized_noised_cond + (1.0 - rescale) * noised_cond`.
        - Normalization attempts to preserve the mean and standard deviation of the original conditioning after noise has been added. This can be useful when using high `noise_scale` values, as it can help prevent the output from becoming too chaotic by "taming" the noised conditioning.
        - If the standard deviation of the noised tensor is too small (close to zero), the rescaling step might be skipped to prevent division by zero or NaN/Inf values, and a warning will be printed.

- **`start_step`**:
    - **Type**: `INT`
    - **Default**: `0`
    - **Min/Max**: `0` / `10000`
    - **Purpose**: This parameter, along with `total_steps`, influences how the noise scaling schedule (`t` value for `t1`/`t2` comparison) is calculated. It represents the starting step of the current sampling process.
    - **Practical Use**:
        - If `start_step` is greater than or equal to `total_steps`, the algorithm falls back to using the sampler's timestep value directly (normalized from `0` to `999`). This can be less predictable as sampler schedules are not always linear.
        - Typically, this might be used in scenarios like img2img or other workflows where sampling doesn't start from step 0.
        - For standard text-to-image generation, leaving it at `0` (with `total_steps` also at `0` or matching the sampler's steps) is common.

- **`total_steps`**:
    - **Type**: `INT`
    - **Default**: `0`
    - **Min/Max**: `0` / `10000`
    - **Purpose**: This parameter, along with `start_step`, influences the noise scaling schedule. It represents the total number of steps in the sampling process.
    - **Practical Use**:
        - If `total_steps` is `0` (or less than `start_step`), the node uses the sampler's actual timestep, which can be non-linear.
        - If you want a linear progression of `t` from `0` to `1` over the sampling steps, set `start_step` to `0` and `total_steps` to the number of steps your KSampler is configured for. This provides more predictable control over the `t1`/`t2` transitions.

- **`apply_to`**:
    - **Type**: `STRING` (Dropdown: `both`, `cond`, `uncond`)
    - **Default**: `both`
    - **Purpose**: Determines whether the CADS noise is applied to the positive conditioning (`cond`), negative conditioning (`uncond`), or `both`.
    - **Practical Use**:
        - `both`: Applies noise to both positive and negative prompts. This is the default and often a good starting point.
        - `cond`: Applies noise only to the positive prompt. This can be useful if you want to introduce variety related to your main subject/style but keep the negative prompt's influence more stable.
        - `uncond`: Applies noise only to the negative prompt. This might be used to introduce more variation in what's being avoided, potentially leading to unexpected positive outcomes.

- **`key`**:
    - **Type**: `STRING` (Dropdown: `both`, `y`, `c_crossattn`)
    - **Default**: `y`
    - **Purpose**: Specifies which part of the conditioning dictionary the noise should be applied to.
        - `y`: This refers to the primary conditioning tensor, often related to CLIP's text embeddings. This is the current default and generally makes the most sense for text-to-image models.
        - `c_crossattn`: This refers to the conditioning tensor used for cross-attention layers. Applying noise here was the previous default behavior.
        - `both`: Applies noise to both `y` and `c_crossattn` if they exist in the conditioning dictionary.
    - **Practical Use**:
        - For most users, the default `y` is recommended.
        - Experimenting with `c_crossattn` or `both` might yield different results, but `y` is considered more aligned with the intended application of noise to the main conditioning signal.

- **`noise_type`**:
    - **Type**: `STRING` (Dropdown: `normal`, `uniform`, `exponential`)
    - **Default**: `normal`
    - **Purpose**: Determines the probability distribution of the generated noise.
    - **Practical Use**:
        - `normal`: Samples noise from a normal (Gaussian) distribution. This is a common choice for noise in diffusion models.
        - `uniform`: Samples noise from a uniform distribution (values are equally likely within a range).
        - `exponential`: Samples noise from an exponential distribution.
        - The choice of noise distribution can subtly alter the characteristics of the variations introduced by CADS. `normal` is a good default, but experimentation with other types might be interesting for specific effects.

- **`seed`**:
    - **Type**: `INT`
    - **Default**: `-1`
    - **Min/Max**: `-1` / `2**32`
    - **Purpose**: Sets the seed for the random noise generator used by CADS.
    - **Practical Use**:
        - If `seed` is `-1` (the default), CADS will use ComfyUI's global seed. This means the noise pattern will be consistent with the main generation seed, making generations reproducible.
        - If you set a specific integer value (e.g., `0`, `123`, etc.), CADS will use its own separate random seed. This allows you to have a consistent noise pattern from CADS even if you change the main generation seed, or vice-versa. This can be useful for isolating the effect of CADS noise.
        - If `start_step` is used and a new sampling pass is detected (current sigma is greater than last sigma), the CADS internal generator is re-seeded.

## How CADS Modifies the Model

The CADS node works by wrapping the existing model's `model_function_wrapper`. This means it intercepts the function that the sampler calls at each step. Within this wrapper, CADS:
1. Calculates `gamma`, a scaling factor for the original conditioning, based on the current timestep (`t`), `t1`, and `t2`.
    - If `t <= t1`, `gamma = 1.0` (original conditioning is fully preserved, no noise effect).
    - If `t >= t2`, `gamma = 0.0` (original conditioning is scaled by 0, noise effect is maximal).
    - If `t1 < t < t2`, `gamma` transitions linearly from `1.0` down to `0.0`.
2. Generates noise according to `noise_type` and `seed`.
3. Modifies the relevant conditioning tensor(s) (selected by `key` and `apply_to`) using the formula:
   `new_cond = sqrt(gamma) * original_cond + noise_scale * sqrt(max(0.0, 1.0 - gamma)) * noise`
4. If `rescale > 0.0`:
    - It calculates the mean and standard deviation of `original_cond`.
    - It standardizes `new_cond`.
    - It then rescales the standardized `new_cond` to match the original mean and std deviation.
    - Finally, it blends this rescaled tensor with the `new_cond` from step 3:
      `final_cond = rescale * rescaled_standardized_cond + (1.0 - rescale) * new_cond`
    - Otherwise, `final_cond = new_cond`.
5. Passes the modified conditioning to the original model function (or the next wrapper in the chain).

The node attempts to preserve any existing wrappers, so it should generally be applied *after* other nodes that modify the model's unet wrapper function.

## Example Usage

The most straightforward way to see CADS in action is to use the example workflow provided in the repository: `example_workflows/CADScompare.json`. You can load this workflow into ComfyUI.

**Basic Workflow Structure:**

1.  **Load Checkpoint**: Use a `CheckpointLoaderSimple` node.
2.  **Encode Prompts**: Use `CLIPTextEncode` nodes for positive and negative prompts.
3.  **CADS Node**:
    *   Connect the `MODEL` output from the Checkpoint Loader to the `model` input of the `CADS` node.
    *   Adjust `noise_scale`, `t1`, `t2`, and other CADS parameters as desired.
4.  **KSampler (with CADS)**:
    *   Connect the `MODEL` output from the `CADS` node to the `model` input of a `KSampler`.
    *   Connect your encoded prompts and an `EmptyLatentImage`.
5.  **KSampler (without CADS - for comparison)**:
    *   Optionally, add a second `KSampler` that takes the original `MODEL` output directly from the Checkpoint Loader. This allows for a direct comparison.
6.  **VAEDecode and PreviewImage**: Decode the latent outputs from the KSamplers to view the generated images.

**Tips for Experimentation:**

*   **Start Simple**: Begin with a moderate `noise_scale` (e.g., 0.1 - 0.3), default `t1` (0.6) and `t2` (0.9), and `rescale` at 0.
*   **Isolate Changes**: Modify one CADS parameter at a time to understand its specific effect.
*   **Use Fixed Seeds**: When comparing the effect of different CADS settings, use the same seed for the KSampler and, if you want consistent CADS noise, set a specific integer for the CADS `seed` parameter. If you want CADS noise to vary with the main seed, leave the CADS `seed` at -1.
*   **Iterate on `noise_scale`**: This is often the most impactful parameter. Gradually increase or decrease it.
*   **Adjust `t1` and `t2`**:
    *   To make noise effective for longer (more of the early/mid diffusion steps): decrease `t2` and `t1`. For example, `t1=0.3`, `t2=0.7`.
    *   To make noise effective for shorter (more of the later diffusion steps): increase `t1` and `t2`. For example, `t1=0.8`, `t2=0.95`. (Note: `t1` should remain less than `t2`).
*   **Experiment with `rescale`**: If high `noise_scale` values produce overly chaotic results, try increasing `rescale` (e.g., 0.25, 0.5). This can help to "tame" the noise while still allowing for its variational benefits.
*   **`apply_to` and `key`**: For most standard use cases, `apply_to: both` and `key: y` are good starting points. Changing these can be useful for more advanced experimentation or troubleshooting.
*   **`total_steps` for Predictable Scheduling**: If you want the `t1`/`t2` schedule to be perfectly aligned with your sampler's progress, set `total_steps` in the CADS node to match the `steps` parameter in your KSampler, and leave `start_step` at 0. If `total_steps` is 0, CADS uses the sampler's sigma values, which might not progress linearly.

## Known Bugs and Limitations

*   **Implementation Correctness**: The author notes that the implementation might not be 100% correct according to the original CADS paper, as the exact mathematical application of noise in the context of image models was not entirely clear. Results might differ from other CADS implementations (like the A1111 version). However, it is still found to be useful for adding variety.
*   **Noise Application Point (`key` parameter)**:
    *   Previously, noise was applied to `c_crossattn`.
    *   It is now applied by default to `y` (primary conditioning tensor), which is believed to make more sense. The `key` parameter allows users to revert to the old behavior or apply to both if desired.
*   **NaNs with `rescale`**: If `rescale` is active and the standard deviation of the noised tensor (before normalization) is very close to zero, the normalization process could lead to NaN (Not a Number) or Inf (Infinity) values. The code includes a check to prevent division by a very small standard deviation and will print a warning, falling back to the non-rescaled noised tensor. Similarly, if the *original* standard deviation (used for rescaling back) is too small, it will also adjust behavior. If NaNs are detected in the `y_rescaled_to_original_stats` tensor, it will also fall back.
*   **Interaction with Other Wrappers**: While the node attempts to preserve existing model function wrappers, complex interactions with multiple wrappers could lead to unexpected behavior. It's generally recommended to apply CADS *after* other similar model-modifying nodes.
