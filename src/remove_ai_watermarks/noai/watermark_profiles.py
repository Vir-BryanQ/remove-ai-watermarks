"""Watermark removal model profiles and the default strength.

Pure configuration and lookup functions with no ML dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# The SDXL-native canny ControlNet used by the ``controlnet`` pipeline. The
# ControlNet is an add-on to the SDXL base checkpoint (DEFAULT_MODEL_ID), not a
# separate base model, so both the ``default`` and ``controlnet`` profiles load
# the same base weights and share the same vendor-adaptive strength.
CONTROLNET_CANNY_MODEL = "xinsir/controlnet-canny-sdxl-1.0"

# Vendor-adaptive default denoising strength for the SDXL img2img scrub, overridable
# from the CLI (`--strength`). The right strength depends on which vendor's SynthID is
# present, detected from the C2PA issuer (metadata.synthid_source). Oracle-verified
# controlled study (2026-06-01, clean v0.8.6, per-image openai.com/verify or Gemini-app
# verdict; see docs/synthid.md section 2.2):
#   - OpenAI gpt-image: removed at 0.05 across 1024-1600 (n=4), resolution-independent.
#     OPENAI_STRENGTH 0.10 = the 0.05 floor plus a 2x margin (keeps quality high).
#   - Google Gemini: removed at 0.15 on the capped-1536 path (n=4); 0.05/0.10 do NOT
#     clear. GEMINI_STRENGTH 0.15. CAVEAT: 0.15 was validated only on
#     `--max-resolution 1536`; native 2816 (the default path) was not locally
#     measurable (OOM on Apple Silicon) and may need more -- pending GPU validation on
#     the raiw.cc backend. If a native large Gemini still verifies positive at 0.15,
#     raise `--strength`.
#   - Unknown vendor (metadata stripped, or non-OpenAI/Google C2PA): UNKNOWN_STRENGTH
#     0.15, the safe middle that clears both vendors at the tested resolutions.
# The dominant factor is VENDOR, not resolution: Google's SynthID is ~3x more robust
# than OpenAI's. The ``controlnet`` pipeline shares these strengths (same SDXL base; the
# canny ControlNet only preserves structure, the strength still drives removal).
OPENAI_STRENGTH = 0.10
GEMINI_STRENGTH = 0.15
UNKNOWN_STRENGTH = 0.15
# Backwards-compatible alias: the vendor-unknown default (what a caller gets without a
# detected vendor). Kept as DEFAULT_STRENGTH for existing references.
DEFAULT_STRENGTH = UNKNOWN_STRENGTH

# Detected-vendor -> default strength. Vendor strings come from `vendor_for_strength`.
_VENDOR_STRENGTH = {"openai": OPENAI_STRENGTH, "google": GEMINI_STRENGTH}


def resolve_strength(strength: float | None, vendor: str | None = None) -> float:
    """Resolve the denoising strength, applying the vendor default when unset.

    ``None`` means "the user did not pass ``--strength``", which resolves
    **vendor-adaptively**: ``vendor`` (``"openai"`` / ``"google"`` / None, from
    ``vendor_for_strength``) selects ``OPENAI_STRENGTH`` / ``GEMINI_STRENGTH`` /
    ``UNKNOWN_STRENGTH``. An explicit value always wins (including ``0.0`` -- the check
    is ``is None``, not falsiness). The ``default`` and ``controlnet`` profiles share
    the same SDXL base (the ControlNet only preserves structure), so the default does
    NOT depend on the profile. Shared by the CLI (for display) and the engine (for
    execution) so the two never disagree -- both must pass the SAME ``vendor``.
    """
    if strength is not None:
        return strength
    return _VENDOR_STRENGTH.get(vendor or "", UNKNOWN_STRENGTH)


def vendor_for_strength(image_path: Path) -> Literal["openai", "google"] | None:
    """Detect the SynthID vendor for strength selection: ``"openai"`` / ``"google"`` / None.

    Reads the C2PA SynthID proxy (``metadata.synthid_source``) on the ORIGINAL input,
    so it must run before any pass that strips metadata. When both issuers appear (a
    rare multi-sign anomaly) Google wins -- the more-robust watermark -> safer (higher)
    strength. Returns None when metadata is stripped or the issuer is neither vendor,
    which maps to ``UNKNOWN_STRENGTH``. Lazy-imports ``metadata`` to keep this module
    dependency-light.
    """
    try:
        from remove_ai_watermarks.metadata import synthid_source

        src = (synthid_source(image_path) or "").lower()
    except Exception:  # metadata unreadable -> treat as unknown vendor
        return None
    if "google" in src:
        return "google"
    if "openai" in src:
        return "openai"
    return None


def get_model_id_for_profile(profile: str) -> str:
    """Map CLI model profile names to concrete Hugging Face model IDs.

    Both ``default`` and ``controlnet`` use the SDXL base checkpoint -- the canny
    ControlNet (``CONTROLNET_CANNY_MODEL``) is an add-on loaded on top of it, not a
    separate base model.
    """
    normalized = profile.strip().lower()
    if normalized in ("default", "controlnet"):
        return DEFAULT_MODEL_ID
    raise ValueError(f"Unknown model profile '{profile}'. Use one of: default, controlnet.")
