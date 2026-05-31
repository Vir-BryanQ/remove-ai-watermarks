"""Watermark removal model profiles, the default strength, and profile detection.

Pure configuration and lookup functions with no ML dependencies.
"""

from __future__ import annotations

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CTRLREGEN_MODEL_ID = "yepengliu/ctrlregen"

# Single default denoising strength for the SDXL img2img scrub, overridable from
# the CLI (`--strength`). Raised from the old 0.04/0.05 because that no longer
# removes the CURRENT Google SynthID (Nano Banana / Gemini 3): verified 2026-05-30
# via the Gemini "Verify with SynthID" oracle on a real image -- 0.05 still
# detected, 0.10 not detected (OpenAI's SynthID was already cleared at 0.05). 0.10
# keeps the visible change modest while removing both. CAVEAT: confirmed on n=1
# Google + n=1 OpenAI image; broad oracle validation across the corpus is pending
# (different images may need a different strength). At this higher strength small
# text deforms more -- which is exactly why text protection (`_run_region_hires`)
# runs by default. (Fixed LOW/MEDIUM/HIGH presets were removed -- unused; the one
# knob is this default plus the per-call `--strength` override.)
DEFAULT_STRENGTH = 0.10

# CtrlRegen removes watermarks by regenerating from (near) clean Gaussian noise,
# NOT by the light-touch partial-noise img2img the SDXL default uses. The research
# is explicit (CtrlRegen, ICLR 2025, arXiv:2410.05470): partial-noise regeneration
# "struggles with high-perturbation watermarks" because a small noise step "retains"
# watermark information that diffuses back into the output; the fix is to start from
# clean noise. With the StableDiffusionControlNetImg2ImgPipeline that maps to a high
# strength (~1.0 = full noise at the first timestep, structure held by the canny
# ControlNet + DINOv2 IP-Adapter, not by the watermarked latent). So the ctrlregen
# profile must NOT inherit the SDXL 0.10 default -- at 0.10 it loads ControlNet +
# DINOv2-giant and then barely changes the image (a no-op for removal). Tunable via
# `--strength`; lower it to trade removal strength for fidelity (the CtrlRegen+ regime).
CTRLREGEN_DEFAULT_STRENGTH = 1.0


def resolve_strength(strength: float | None, profile: str) -> float:
    """Resolve the denoising strength, applying the profile-specific default when unset.

    ``None`` means "the user did not pass ``--strength``": the SDXL default profile
    resolves to ``DEFAULT_STRENGTH`` (a light SynthID-tuned touch), while ``ctrlregen``
    resolves to ``CTRLREGEN_DEFAULT_STRENGTH`` (clean-noise regeneration). An explicit
    value always wins. Shared by the CLI (for display) and the engine (for execution)
    so the two never disagree.
    """
    if strength is not None:
        return strength
    return CTRLREGEN_DEFAULT_STRENGTH if profile == "ctrlregen" else DEFAULT_STRENGTH


def get_model_id_for_profile(profile: str) -> str:
    """Map CLI model profile names to concrete Hugging Face model IDs."""
    normalized = profile.strip().lower()
    if normalized == "default":
        return DEFAULT_MODEL_ID
    if normalized == "ctrlregen":
        return CTRLREGEN_MODEL_ID
    raise ValueError(f"Unknown model profile '{profile}'. Use one of: default, ctrlregen.")


def detect_model_profile(model_id: str) -> str:
    """Infer model profile from model identifier."""
    if "ctrlregen" in model_id.lower():
        return "ctrlregen"
    return "default"
