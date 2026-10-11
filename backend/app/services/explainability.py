import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("app.explainability")

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available — explainability will use fallback heatmap.")

GRAD_CAM_AVAILABLE = False
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    logger.warning("pytorch-grad-cam not installed. pip install grad-cam to enable GradCAM heatmaps.")


# ──────────────────────────────────────────────────────────────
# GradCAM for BiomedCLIP (RGB branch)
# ──────────────────────────────────────────────────────────────

def _get_biomedclip_target_layers(visual_model):
    """
    Resolves the target conv/attention layer for GradCAM within BiomedCLIP's ViT-B/16 visual backbone.
    We target the last transformer block's layer norm as a proxy since ViTs don't have conv layers.
    Falls back to the full visual module if structure is unexpected.
    """
    try:
        # open_clip ViT: model.visual.transformer.resblocks[-1]
        resblocks = visual_model.transformer.resblocks
        return [resblocks[-1].ln_1]
    except AttributeError:
        pass
    try:
        # alternate open_clip structure
        blocks = list(visual_model.blocks)
        return [blocks[-1].norm1]
    except AttributeError:
        pass
    # Last-resort: try returning the visual model itself
    return [visual_model]


def generate_biomedclip_gradcam(
    img_path: str,
    biomed_clip_service,
    pred_idx: int,
    uploads_dir: str
) -> str:
    """
    Generate a GradCAM heatmap overlay for a BiomedCLIP RGB classification prediction.

    Args:
        img_path:            Absolute path to the input image.
        biomed_clip_service: Loaded BiomedCLIPService instance (must have .model and .head).
        pred_idx:            Predicted class index (int).
        uploads_dir:         Directory to save the heatmap overlay PNG.

    Returns:
        Filename of the saved heatmap PNG (e.g. "gradcam_uuid.png"), or "" on failure.
    """
    base_name = os.path.basename(img_path).split(".")[0]
    heatmap_filename = f"gradcam_{base_name}.png"
    heatmap_path = os.path.join(uploads_dir, heatmap_filename)

    # ── GradCAM path (requires pytorch-grad-cam) ──────────────
    if TORCH_AVAILABLE and GRAD_CAM_AVAILABLE and biomed_clip_service.model is not None:
        try:
            visual = biomed_clip_service.model.visual
            target_layers = _get_biomedclip_target_layers(visual)

            # Wrap model so GradCAM sees a single-tensor output
            class _BiomedWrapper(nn.Module):
                def __init__(self, visual, head):
                    super().__init__()
                    self.visual = visual
                    self.head = head

                def forward(self, x):
                    feats = self.visual(x)
                    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
                    return self.head(feats)

            wrapper = _BiomedWrapper(visual, biomed_clip_service.head)
            wrapper.eval()

            preprocess = biomed_clip_service.preprocess
            with Image.open(img_path) as pil_img:
                input_tensor = preprocess(pil_img).unsqueeze(0)
                # Keep a float32 numpy copy for overlay
                rgb_np = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0

            cam = GradCAM(model=wrapper, target_layers=target_layers)
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W) float

            # Colorize: blue→green→red jet-like overlay
            heatmap_colored = _apply_colormap(grayscale_cam)  # (H, W, 3) uint8
            heatmap_resized = np.array(
                Image.fromarray(heatmap_colored).resize((rgb_np.shape[1], rgb_np.shape[0]))
            ).astype(np.float32) / 255.0

            overlay = (0.55 * rgb_np + 0.45 * heatmap_resized)
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
            Image.fromarray(overlay).save(heatmap_path)
            logger.info(f"GradCAM heatmap saved: {heatmap_path}")
            return heatmap_filename

        except Exception as e:
            logger.error(f"GradCAM (BiomedCLIP) failed: {e}. Falling back to attention heatmap.")

    # ── Attention-based fallback using BiomedCLIP's ViT attention weights ──
    if TORCH_AVAILABLE and biomed_clip_service.model is not None:
        try:
            return _vit_attention_heatmap(
                img_path, biomed_clip_service.model.visual,
                biomed_clip_service.preprocess, heatmap_path, heatmap_filename,
                patch_size=16, img_size=224
            )
        except Exception as e:
            logger.error(f"BiomedCLIP attention heatmap failed: {e}")

    # ── Last resort: synthetic importance heatmap (deterministic) ──
    return _synthetic_heatmap(img_path, heatmap_path, heatmap_filename)


# ──────────────────────────────────────────────────────────────
# DINOv2 Self-Attention Heatmap (Scan branch)
# ──────────────────────────────────────────────────────────────

def generate_dinov2_attention_heatmap(
    img_path: str,
    rad_dino_service,
    uploads_dir: str
) -> str:
    """
    Generate a DINOv2 self-attention heatmap overlay for a radiology scan prediction.
    DINOv2 natively produces rich self-attention maps from the [CLS] token — no GradCAM needed.

    Args:
        img_path:         Absolute path to the input image.
        rad_dino_service: Loaded RadDINOService instance (must have .model).
        uploads_dir:      Directory to save the heatmap overlay PNG.

    Returns:
        Filename of saved heatmap PNG, or "" on failure.
    """
    base_name = os.path.basename(img_path).split(".")[0]
    heatmap_filename = f"gradcam_{base_name}.png"
    heatmap_path = os.path.join(uploads_dir, heatmap_filename)

    if TORCH_AVAILABLE and rad_dino_service.model is not None:
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            with Image.open(img_path) as pil_img:
                rgb_pil = pil_img.convert("RGB")
                input_tensor = transform(rgb_pil).unsqueeze(0)
                rgb_np = np.array(rgb_pil.resize((224, 224))).astype(np.float32) / 255.0

            model = rad_dino_service.model

            # Hook into the last attention block of DINOv2 ViT-S/14
            attention_maps = {}

            def _attn_hook(module, input, output):
                attention_maps["attn"] = output.detach()

            # DINOv2 ViT-S/14 last block: model.blocks[-1].attn
            hook = model.blocks[-1].attn.register_forward_hook(_attn_hook)

            with torch.no_grad():
                _ = model(input_tensor)

            hook.remove()

            if "attn" not in attention_maps:
                raise ValueError("Attention map not captured by hook.")

            # attn shape: (1, num_heads, num_patches+1, num_patches+1)
            attn = attention_maps["attn"][0]  # (heads, seq, seq)
            # CLS token attends to all patches: take CLS row, average over heads
            cls_attn = attn[:, 0, 1:].mean(0).cpu().numpy()  # (num_patches,)

            # DINOv2 ViT-S/14 patch_size=14, so 224/14 = 16 patches per side
            grid_size = 224 // 14  # = 16
            cls_attn = cls_attn[:grid_size * grid_size].reshape(grid_size, grid_size)

            # Normalize to [0, 1]
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

            # Upscale from 16×16 → 224×224
            attn_img = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize(
                (224, 224), resample=Image.BILINEAR
            )
            attn_np = np.array(attn_img).astype(np.float32) / 255.0

            heatmap_colored = _apply_colormap(attn_np).astype(np.float32) / 255.0
            overlay = (0.55 * rgb_np + 0.45 * heatmap_colored)
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

            # Resize back to original image size
            with Image.open(img_path) as orig:
                orig_size = orig.size  # (W, H)
            Image.fromarray(overlay).resize(orig_size, Image.LANCZOS).save(heatmap_path)
            logger.info(f"DINOv2 attention heatmap saved: {heatmap_path}")
            return heatmap_filename

        except Exception as e:
            logger.error(f"DINOv2 attention heatmap failed: {e}. Falling back.")

    return _synthetic_heatmap(img_path, heatmap_path, heatmap_filename)


# ──────────────────────────────────────────────────────────────
# Shared Utilities
# ──────────────────────────────────────────────────────────────

def _vit_attention_heatmap(img_path, visual_model, preprocess, heatmap_path, heatmap_filename, patch_size=16, img_size=224):
    """
    Generic ViT attention heatmap for open_clip ViT-B/16 models.
    Hooks into the last transformer block's attention module.
    """
    with Image.open(img_path) as pil_img:
        input_tensor = preprocess(pil_img).unsqueeze(0)
        rgb_np = np.array(pil_img.convert("RGB").resize((img_size, img_size))).astype(np.float32) / 255.0

    attention_maps = {}

    def _hook(module, inp, out):
        attention_maps["attn"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()

    # open_clip ViT structure
    last_block = visual_model.transformer.resblocks[-1]
    hook = last_block.attn.register_forward_hook(_hook)

    with torch.no_grad():
        _ = visual_model(input_tensor)
    hook.remove()

    if "attn" not in attention_maps:
        raise ValueError("Attention map not captured.")

    attn = attention_maps["attn"]
    if attn.dim() == 4:  # (batch, heads, seq, seq)
        cls_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy()
    elif attn.dim() == 3:  # (seq, batch, dim) — open_clip MultiheadAttention format
        cls_attn = attn[0, 0, 1:].cpu().numpy()
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size
    cls_attn = cls_attn[:num_patches].reshape(grid_size, grid_size)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    attn_img = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    attn_np = np.array(attn_img).astype(np.float32) / 255.0

    heatmap_colored = _apply_colormap(attn_np).astype(np.float32) / 255.0
    overlay = np.clip((0.55 * rgb_np + 0.45 * heatmap_colored) * 255, 0, 255).astype(np.uint8)

    with Image.open(img_path) as orig:
        orig_w, orig_h = orig.size
    Image.fromarray(overlay).resize((orig_w, orig_h), Image.LANCZOS).save(heatmap_path)
    return heatmap_filename


def _synthetic_heatmap(img_path: str, heatmap_path: str, heatmap_filename: str) -> str:
    """
    Deterministic synthetic heatmap based on image gradient magnitude.
    Used only when all model-based explainability paths fail.
    Produces a visually plausible (but not model-driven) importance map.
    """
    try:
        with Image.open(img_path) as pil_img:
            gray = np.array(pil_img.convert("L").resize((224, 224))).astype(np.float32)
            rgb_np = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
            orig_size = pil_img.size

        # Sobel-like gradient magnitude as saliency proxy
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        grad_mag = np.sqrt(gx**2 + gy**2)
        # Gaussian blur to smooth
        from PIL import ImageFilter
        grad_img = Image.fromarray(grad_mag.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=8))
        grad_np = np.array(grad_img).astype(np.float32)
        grad_np = (grad_np - grad_np.min()) / (grad_np.max() - grad_np.min() + 1e-8)

        heatmap_colored = _apply_colormap(grad_np).astype(np.float32) / 255.0
        overlay = np.clip((0.6 * rgb_np + 0.4 * heatmap_colored) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(overlay).resize(orig_size, Image.LANCZOS).save(heatmap_path)
        logger.info(f"Synthetic gradient heatmap saved: {heatmap_path}")
        return heatmap_filename
    except Exception as e:
        logger.error(f"Synthetic heatmap failed: {e}")
        return ""


def _apply_colormap(gray_2d: np.ndarray) -> np.ndarray:
    """
    Converts a 2D float [0,1] array to a jet-like RGB colormap (H, W, 3) uint8.
    Pure NumPy implementation — no matplotlib required.
    """
    h, w = gray_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Jet colormap approximation
    # R: rises 0→1 in [0.5, 0.75], stays at 1 in [0.75, 1.0]
    # G: rises 0→1 in [0.25, 0.5], falls 1→0 in [0.5, 0.75]
    # B: stays 1 in [0.0, 0.25], falls 1→0 in [0.25, 0.5]
    v = gray_2d
    rgb[..., 0] = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)  # R
    rgb[..., 1] = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)  # G
    rgb[..., 2] = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)  # B

    return (rgb * 255).astype(np.uint8)
