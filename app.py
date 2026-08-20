# -*- coding: utf-8 -*-
"""
BodyError // Loop507
=====================
Trasforma una foto in un video "Body Error Realistico": scomposizione
anatomica progressiva, guidata dall'energia di una traccia audio opzionale.
Pure DSP / algoritmico - nessuna rete neurale, nessuna AI generativa.

Loop507 protocol:
- py_compile + pyflakes zero warning
- report bilingue IT/EN
- session_state per persistenza download
- seed system per riproducibilita'

Deploy: sostituisci il file su GitHub -> commit -> "Reboot app" dal Manage panel.
"""

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import soundfile as sf
import streamlit as st

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


# ---------------------------------------------------------------------------
# COSTANTI / REGISTRY STILI
# ---------------------------------------------------------------------------

APP_TITLE = "BodyError // Loop507"

STYLE_MELT = "plastination_melt"
STYLE_VORONOI = "voronoi_fracture"
STYLE_REACTION = "reaction_diffusion_bloom"
STYLE_DEPTH = "depth_peel"
STYLE_CAPILLARY = "capillary_bleed"
STYLE_FLUID = "fluid_melt"

STYLE_LABELS = {
    STYLE_MELT: "Plastination Melt",
    STYLE_VORONOI: "Voronoi Fracture",
    STYLE_REACTION: "Reaction-Diffusion Bloom",
    STYLE_DEPTH: "Depth Peel",
    STYLE_CAPILLARY: "Capillary Bleed",
    STYLE_FLUID: "Fluid Melt",
}

IMPLEMENTED_STYLES = {
    STYLE_MELT, STYLE_VORONOI, STYLE_REACTION, STYLE_DEPTH,
    STYLE_CAPILLARY, STYLE_FLUID,
}

ASPECT_PRESETS = {
    "16:9  (1280x720)": (1280, 720),
    "9:16  (720x1280)": (720, 1280),
    "1:1   (720x720)": (720, 720),
}

MAX_DURATION_SEC = 300  # 5 minuti, con avviso sul tempo di rendering


# ---------------------------------------------------------------------------
# UTILITY IMMAGINE
# ---------------------------------------------------------------------------

def load_image_fit_aspect(path, target_w, target_h):
    """
    Carica l'immagine e la adatta alla risoluzione target tramite center-crop
    (mantiene l'aspect ratio richiesto senza deformare il soggetto) + resize.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Immagine non leggibile / Image could not be read")

    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # sorgente piu' larga: crop orizzontale
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        # sorgente piu' alta: crop verticale
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]

    interp = cv2.INTER_AREA if img.shape[0] > target_h else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (target_w, target_h), interpolation=interp)
    return img.astype(np.float32) / 255.0


def build_subject_mask(img):
    gray0 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    bg_val = float(np.median(gray0[:20, :20]))
    mask = (np.abs(gray0.astype(np.float32) - bg_val) > 12).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = np.clip(mask * 1.4, 0, 1)
    return mask


def clinical_grade(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = np.stack([gray] * 3, axis=-1)
    img = img * 0.55 + gray3 * 0.45
    tint = np.array([1.08, 1.03, 0.88], dtype=np.float32)
    img = np.clip(img * tint, 0, 1)
    img = np.clip((img - 0.5) * 1.15 + 0.5, 0, 1)

    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - 0.35 * (dist / max_dist) ** 2
    vignette = np.clip(vignette, 0.5, 1.0)[..., None]
    img = img * vignette
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# ANALISI AUDIO -> ENVELOPE DI INTENSITA'
# ---------------------------------------------------------------------------

def get_audio_duration(path):
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def analyze_audio(path, target_fps, duration_sec):
    """
    Estrae un envelope di energia normalizzato [0,1] campionato a target_fps,
    piu' i frame corrispondenti ai beat, per la micro-variazione dei seed.
    """
    y, sr = librosa.load(path, sr=22050, mono=True, duration=duration_sec)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

    total_frames = int(duration_sec * target_fps)
    video_times = np.linspace(0, duration_sec, total_frames)

    # interpolazione RMS sui tempi video + normalizzazione robusta
    envelope = np.interp(video_times, rms_times, rms)
    p5, p95 = np.percentile(envelope, [5, 95])
    envelope = np.clip((envelope - p5) / max(p95 - p5, 1e-6), 0, 1)
    # smoothing leggero per evitare tremolio frame-a-frame
    kernel = np.ones(5) / 5
    envelope = np.convolve(envelope, kernel, mode="same")

    beat_video_frames = sorted(set(
        int(np.clip(bt / duration_sec * total_frames, 0, total_frames - 1))
        for bt in beat_times if bt <= duration_sec
    ))

    bpm_value = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return envelope, beat_video_frames, bpm_value


def synthetic_envelope(target_fps, duration_sec):
    """
    Fallback ADSR-style se non c'e' audio: attacco lento, sustain con
    micro-variazione, release parziale verso la fine. Niente crescita lineare.
    """
    total_frames = int(duration_sec * target_fps)
    t = np.linspace(0, 1, total_frames)

    attack_end = 0.25
    decay_end = 0.4
    sustain_end = 0.85

    env = np.zeros(total_frames)
    for i, tv in enumerate(t):
        if tv < attack_end:
            env[i] = tv / attack_end
        elif tv < decay_end:
            local = (tv - attack_end) / (decay_end - attack_end)
            env[i] = 1.0 - 0.25 * local
        elif tv < sustain_end:
            local = (tv - decay_end) / (sustain_end - decay_end)
            env[i] = 0.75 + 0.2 * np.sin(local * np.pi * 3) * 0.5 + 0.1 * local
        else:
            local = (tv - sustain_end) / (1 - sustain_end)
            env[i] = 0.9 + 0.1 * local

    env = np.clip(env, 0, 1)
    # beat sintetici regolari (ogni ~0.5s) solo per le micro-variazioni di seed
    beat_frames = list(range(0, total_frames, max(int(target_fps * 0.5), 1)))
    return env, beat_frames, None


# ---------------------------------------------------------------------------
# STILE: PLASTINATION MELT
# ---------------------------------------------------------------------------

def melt_warp_step(img, subject_mask, strength, melt_bias=1.6):
    h, w = img.shape[:2]
    xx_base, yy_base = np.meshgrid(np.arange(w).astype(np.float32),
                                    np.arange(h).astype(np.float32))

    gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    gx_n = gx / mag
    gy_n = gy / mag

    edge_strength = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    edge_strength = cv2.GaussianBlur(edge_strength, (41, 41), 0)

    flow_x = gx_n * edge_strength * strength * subject_mask
    flow_y = (gy_n * 0.25 + melt_bias) * edge_strength * strength * subject_mask

    new_x = np.clip(xx_base + flow_x, 0, w - 1)
    new_y = np.clip(yy_base + flow_y, 0, h - 1)

    return cv2.remap(img, new_x, new_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def layer_peel(img, n_layers, offset_step):
    gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    h, w = img.shape[:2]
    result = img.copy()

    for layer in range(1, n_layers + 1):
        offset = layer * offset_step
        m = np.float32([[1, 0, offset * 0.4], [0, 1, offset]])
        shifted_edges = cv2.warpAffine(edges, m, (w, h))

        layer_color = np.array([0.25 - layer * 0.03, 0.05, 0.35 + layer * 0.05],
                                dtype=np.float32)
        mask = (shifted_edges > 0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
        alpha = 0.35 / layer

        color_layer = np.ones_like(img) * layer_color
        result = result * (1 - mask * alpha) + color_layer * (mask * alpha)

    return result


def render_melt(base_img, subject_mask, envelope, beat_frames, seed,
                 max_strength, max_layers, layer_offset):
    rng = np.random.default_rng(seed)
    total_frames = len(envelope)
    accumulated = base_img.copy()
    frames = []

    for f in range(total_frames):
        e = envelope[f]
        prev_e = envelope[f - 1] if f > 0 else 0.0
        delta = max(e - prev_e, 0.0) + 0.02  # spinta minima costante per non stagnare

        step_strength = float((max_strength / total_frames) * (0.5 + 3.0 * delta))
        if f in beat_frames:
            step_strength *= 1.6  # burst sul beat
            _ = rng.uniform(-1, 1)  # micro-jitter riservato a estensioni future

        accumulated = melt_warp_step(accumulated, subject_mask, step_strength)

        current_layers = max(1, int(1 + e * (max_layers - 1)))
        current_offset = layer_offset * (0.3 + 0.7 * e)
        frame = layer_peel(accumulated, current_layers, current_offset)
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# STILE: VORONOI FRACTURE
# ---------------------------------------------------------------------------

def seed_points_on_edges(gray, n_points, rng):
    edges = cv2.Canny(gray, 50, 130)
    ys, xs = np.where(edges > 0)
    h, w = gray.shape
    if len(xs) < n_points:
        extra = n_points - len(xs)
        xs = np.concatenate([xs, rng.integers(0, w, extra)])
        ys = np.concatenate([ys, rng.integers(0, h, extra)])
    idx = rng.choice(len(xs), n_points, replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    rand_pts = rng.integers(0, [w, h], size=(max(n_points // 5, 4), 2)).astype(np.float32)
    pts = np.concatenate([pts, rand_pts], axis=0)
    return pts


def build_cell_labels(shape, points):
    from scipy.spatial import cKDTree
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(points)
    _, labels = tree.query(coords, k=1)
    return labels.reshape(h, w)


def voronoi_fracture_frame(img, intensity, n_points, rng):
    h, w = img.shape[:2]
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    points = seed_points_on_edges(gray_blur, n_points, rng)
    labels = build_cell_labels((h, w), points)
    n_cells = len(points)
    cx, cy = w / 2, h / 2

    disp_x = np.zeros(n_cells, dtype=np.float32)
    disp_y = np.zeros(n_cells, dtype=np.float32)
    for i, (px, py) in enumerate(points):
        dx, dy = px - cx, py - cy
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        push = (dist / max(w, h)) * 1.4
        disp_x[i] = (dx / dist) * push * intensity * 55
        disp_y[i] = (dy / dist) * push * intensity * 55 + 0.9 * intensity * 40

    disp_x += rng.uniform(-8, 8, n_cells) * intensity
    disp_y += rng.uniform(-4, 14, n_cells) * intensity

    result = np.zeros_like(img)
    result[..., 0] = 0.03
    result[..., 1] = 0.02
    result[..., 2] = 0.05

    for i in range(n_cells):
        cell_mask = (labels == i)
        if not np.any(cell_mask):
            continue
        ys, xs = np.where(cell_mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

        shift_x = int(round(disp_x[i]))
        shift_y = int(round(disp_y[i]))
        ny0, ny1 = y0 + shift_y, y1 + shift_y
        nx0, nx1 = x0 + shift_x, x1 + shift_x

        src_y0, src_y1, src_x0, src_x1 = y0, y1, x0, x1
        if ny0 < 0:
            src_y0 -= ny0
            ny0 = 0
        if nx0 < 0:
            src_x0 -= nx0
            nx0 = 0
        if ny1 > h:
            src_y1 -= (ny1 - h)
            ny1 = h
        if nx1 > w:
            src_x1 -= (nx1 - w)
            nx1 = w
        if ny1 <= ny0 or nx1 <= nx0:
            continue

        sub_mask = cell_mask[src_y0:src_y1, src_x0:src_x1]
        sub_img = img[src_y0:src_y1, src_x0:src_x1]
        dest = result[ny0:ny1, nx0:nx1]
        dest[sub_mask] = sub_img[sub_mask]
        result[ny0:ny1, nx0:nx1] = dest

    moved_mask = (result.sum(axis=-1) > 0.2).astype(np.uint8) * 255
    piece_edges = cv2.Canny(moved_mask, 50, 150)
    piece_edges = cv2.dilate(piece_edges, np.ones((2, 2), np.uint8))
    result[piece_edges > 0] *= 0.2

    return np.clip(result, 0, 1)


def render_voronoi(base_img, envelope, beat_frames, seed, max_intensity, n_points):
    rng_master = np.random.default_rng(seed)
    total_frames = len(envelope)
    frames = []
    beat_set = set(beat_frames)
    current_seed_offset = 0

    for f in range(total_frames):
        if f in beat_set:
            current_seed_offset += 1
        local_rng = np.random.default_rng(seed + current_seed_offset)

        e = float(envelope[f])
        intensity = 0.15 + max_intensity * e
        frame = voronoi_fracture_frame(base_img, intensity, n_points, local_rng)
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    _ = rng_master  # riservato per estensioni (variazione palette globale)
    return frames


# ---------------------------------------------------------------------------
# STILE: REACTION-DIFFUSION BLOOM (Gray-Scott model)
# ---------------------------------------------------------------------------

def rd_step(u, v, du=0.16, dv=0.08, feed=0.035, kill=0.065, iterations=1):
    for _ in range(iterations):
        lu = cv2.Laplacian(u, cv2.CV_32F)
        lv = cv2.Laplacian(v, cv2.CV_32F)
        uvv = u * v * v
        u = u + (du * lu - uvv + feed * (1 - u))
        v = v + (dv * lv + uvv - (kill + feed) * v)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
    return u, v


def render_reaction_diffusion(base_img, subject_mask, envelope, beat_frames, seed,
                               max_intensity):
    rng = np.random.default_rng(seed)
    h, w = base_img.shape[:2]

    sim_scale = 0.25
    sh, sw = max(int(h * sim_scale), 24), max(int(w * sim_scale), 24)
    mask_small = cv2.resize(subject_mask, (sw, sh))

    gray_small = cv2.resize(
        cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), (sw, sh))
    edges_small = cv2.Canny(gray_small, 50, 130).astype(np.float32) / 255.0
    seed_prob = edges_small * mask_small

    u = np.ones((sh, sw), dtype=np.float32)
    v = np.zeros((sh, sw), dtype=np.float32)

    ys, xs = np.where(seed_prob > 0.3)
    if len(xs) == 0:
        ys, xs = np.array([sh // 2]), np.array([sw // 2])
    n_seeds = min(18, len(xs))
    idx = rng.choice(len(xs), n_seeds, replace=False)
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        v[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 1.0
        u[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 0.5

    xx_base, yy_base = np.meshgrid(np.arange(w).astype(np.float32),
                                    np.arange(h).astype(np.float32))
    bloom_color = np.array([0.05, 0.12, 0.05], dtype=np.float32)  # BGR verde scuro/muschio

    frames = []
    total_frames = len(envelope)
    for f in range(total_frames):
        e = float(envelope[f])
        iters = 1 + int(e * 4)
        u, v = rd_step(u, v, iterations=iters)

        if f in beat_frames:
            iy, ix = int(rng.integers(0, sh)), int(rng.integers(0, sw))
            v[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = 1.0

        v_big = cv2.resize(v, (w, h), interpolation=cv2.INTER_CUBIC)
        v_big = np.clip(v_big, 0, 1) * subject_mask

        alpha = v_big[..., None] * (0.55 + 0.3 * max_intensity * e)
        frame = base_img * (1 - alpha) + bloom_color * alpha

        gx = cv2.Sobel(v_big, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(v_big, cv2.CV_32F, 0, 1, ksize=5)
        disp = float(10.0 * e)
        new_x = np.clip(xx_base + gx * disp, 0, w - 1)
        new_y = np.clip(yy_base + gy * disp, 0, h - 1)
        frame = cv2.remap(frame, new_x, new_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# STILE: DEPTH PEEL (pseudo-profondita' da shading + sfogliatura a strati)
# ---------------------------------------------------------------------------

def pseudo_depth_map(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth = cv2.GaussianBlur(gray, (25, 25), 0)
    depth = 1.0 - depth  # zone in ombra trattate come piu' profonde
    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    return depth


def depth_peel_frame(img, subject_mask, depth, n_bands, band_offset, intensity):
    h, w = img.shape[:2]
    result = img.copy()

    for b in range(1, n_bands + 1):
        lo = (b - 1) / n_bands
        hi = b / n_bands
        band_mask = ((depth >= lo) & (depth < hi)).astype(np.float32) * subject_mask

        offset = float(b * band_offset * intensity)
        m = np.float32([[1, 0, 0], [0, 1, offset]])
        shifted_img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        shifted_mask = cv2.warpAffine(band_mask, m, (w, h))

        tint = np.array([0.15 - b * 0.01, 0.05, 0.25 + b * 0.03], dtype=np.float32)
        tinted = shifted_img * 0.7 + tint * 0.3

        alpha = shifted_mask[..., None] * 0.5
        result = result * (1 - alpha) + tinted * alpha

    return result


def render_depth_peel(base_img, subject_mask, envelope, beat_frames, seed, max_intensity):
    depth = pseudo_depth_map(base_img)
    rng = np.random.default_rng(seed)
    frames = []
    total_frames = len(envelope)

    for f in range(total_frames):
        e = float(envelope[f])
        n_bands = 2 + int(e * 4)
        band_offset = 4.0
        if f in beat_frames:
            band_offset *= 1.3
        intensity = 0.3 + max_intensity * e

        frame = depth_peel_frame(base_img, subject_mask, depth, n_bands, band_offset, intensity)
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    _ = rng  # riservato per jitter futuro sui bordi di banda
    return frames


# ---------------------------------------------------------------------------
# STILE: CAPILLARY BLEED (random-walk vincolato ai bordi ad alto contrasto)
# ---------------------------------------------------------------------------

def render_capillary_bleed(base_img, subject_mask, envelope, beat_frames, seed,
                            max_intensity, n_walkers=40):
    rng = np.random.default_rng(seed)
    h, w = base_img.shape[:2]

    gray = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    gx_n = gx / mag
    gy_n = gy / mag

    edges = cv2.Canny(gray_blur, 50, 130)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        ys, xs = np.array([h // 2]), np.array([w // 2])
    idx = rng.choice(len(xs), min(n_walkers, len(xs)), replace=False)
    walker_pos = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    walker_dir = rng.uniform(-1, 1, walker_pos.shape).astype(np.float32)

    vein_mask = np.zeros((h, w), dtype=np.float32)
    bleed_color = np.array([0.05, 0.02, 0.35], dtype=np.float32)  # BGR rosso scuro

    frames = []
    total_frames = len(envelope)
    max_walkers = n_walkers * 4

    for f in range(total_frames):
        e = float(envelope[f])
        step_size = 1.0 + 3.0 * e

        xi = np.clip(walker_pos[:, 0].astype(int), 0, w - 1)
        yi = np.clip(walker_pos[:, 1].astype(int), 0, h - 1)
        tangent_x = -gy_n[yi, xi]
        tangent_y = gx_n[yi, xi]
        rand_perturb = rng.uniform(-0.4, 0.4, walker_dir.shape).astype(np.float32)
        walker_dir = (0.7 * walker_dir + 0.3 * np.stack([tangent_x, tangent_y], axis=1)
                      + rand_perturb)
        norm = np.linalg.norm(walker_dir, axis=1, keepdims=True) + 1e-6
        walker_dir = walker_dir / norm

        walker_pos = walker_pos + walker_dir * step_size
        walker_pos[:, 0] = np.clip(walker_pos[:, 0], 0, w - 1)
        walker_pos[:, 1] = np.clip(walker_pos[:, 1], 0, h - 1)

        xi2 = walker_pos[:, 0].astype(int)
        yi2 = walker_pos[:, 1].astype(int)
        vein_mask[yi2, xi2] = 1.0

        if f in beat_frames and len(walker_pos) < max_walkers:
            n_new = min(4, n_walkers)
            branch_idx = rng.integers(0, len(walker_pos), n_new)
            new_pos = walker_pos[branch_idx] + rng.uniform(-2, 2, (n_new, 2))
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0, w - 1)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0, h - 1)
            new_dir = rng.uniform(-1, 1, (n_new, 2)).astype(np.float32)
            walker_pos = np.concatenate([walker_pos, new_pos.astype(np.float32)], axis=0)
            walker_dir = np.concatenate([walker_dir, new_dir], axis=0)

        vein_blur = cv2.GaussianBlur(vein_mask, (3, 3), 0)
        vein_blur = np.clip(vein_blur * subject_mask, 0, 1)

        alpha = vein_blur[..., None] * (0.6 + 0.3 * e)
        frame = base_img * (1 - alpha) + bleed_color * alpha
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# STILE: FLUID MELT (mini solver semi-Lagrangiano, stile Stam)
# ---------------------------------------------------------------------------

def semi_lagrangian_advect(field, vel_x, vel_y, dt=1.0):
    h, w = field.shape[:2]
    xx, yy = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))
    back_x = np.clip(xx - vel_x * dt, 0, w - 1)
    back_y = np.clip(yy - vel_y * dt, 0, h - 1)
    return cv2.remap(field, back_x, back_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def render_fluid_melt(base_img, subject_mask, envelope, beat_frames, seed, max_intensity):
    rng = np.random.default_rng(seed)
    h, w = base_img.shape[:2]

    noise_small = rng.uniform(-1, 1, (max(h // 20, 4), max(w // 20, 4))).astype(np.float32)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=15)

    # campo di velocita' turbolento fisso (no auto-advection: evita il feedback
    # che amplificava la deformazione fino a distruggere l'immagine)
    vel_x_base = noise * 0.5 * subject_mask
    vel_y_base = noise * 0.3 * subject_mask

    accumulated = base_img.copy()
    frames = []
    total_frames = len(envelope)

    for f in range(total_frames):
        e = float(envelope[f])
        gravity = float(0.4 + 0.6 * e)

        vx_frame = np.clip(vel_x_base, -1.5, 1.5)
        vy_frame = np.clip(vel_y_base + gravity * subject_mask, -1.8, 1.8)

        if f in beat_frames:
            vy_frame = vy_frame * 1.3

        vx_frame = cv2.GaussianBlur(vx_frame, (5, 5), 0)
        vy_frame = cv2.GaussianBlur(vy_frame, (5, 5), 0)

        # spostamento per-frame come budget totale ripartito sulla durata,
        # stesso principio di scaling usato in Plastination Melt
        per_frame_dt = float((max_intensity / total_frames) * (0.5 + 2.5 * e))
        accumulated = semi_lagrangian_advect(accumulated, vx_frame, vy_frame, dt=per_frame_dt)

        frame = clinical_grade(accumulated)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    return frames


STYLE_RENDERERS = {
    STYLE_MELT: "melt",
    STYLE_VORONOI: "voronoi",
    STYLE_REACTION: "reaction_diffusion",
    STYLE_DEPTH: "depth_peel",
    STYLE_CAPILLARY: "capillary_bleed",
    STYLE_FLUID: "fluid_melt",
}


# ---------------------------------------------------------------------------
# EXPORT VIDEO + MUX AUDIO
# ---------------------------------------------------------------------------

def write_raw_video(frames, fps, out_path):
    """Scrittura intermedia veloce (mp4v) - non browser-playable, solo staging."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()


def finalize_video(raw_video_path, audio_path, duration_sec, out_path):
    """
    Transcodifica in H.264 / yuv420p (compatibilita' browser per l'anteprima
    st.video e per il download) ed esegue il mux dell'audio se presente,
    nello stesso passaggio ffmpeg.
    """
    if audio_path is not None:
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_video_path,
            "-i", audio_path,
            "-t", str(duration_sec),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "aac", "-shortest",
            "-movflags", "+faststart",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_video_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-movflags", "+faststart",
            out_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# REPORT BILINGUE
# ---------------------------------------------------------------------------

def build_report(style_key, seed, duration_sec, fps, resolution, has_audio, bpm_value):
    style_label = STYLE_LABELS[style_key]
    audio_line_it = f"BPM rilevato: {bpm_value:.1f}" if bpm_value else "Nessun audio: envelope sintetico (ADSR)"
    audio_line_en = f"Detected BPM: {bpm_value:.1f}" if bpm_value else "No audio: synthetic envelope (ADSR)"

    report = f"""
BodyError // Loop507 :: REPORT

[IT]
Stile: {style_label}
Seed: {seed}
Durata: {duration_sec}s @ {fps}fps
Risoluzione: {resolution}
Audio: {"presente" if has_audio else "assente"} — {audio_line_it}
Tecnica: DSP puro (OpenCV/NumPy/SciPy), nessuna rete neurale.
Riferimenti estetici: plastinazione anatomica (von Hagens), iperrealismo
scultoreo clinico (Mueck, Jinks) — reinterpretati come processo algoritmico.

[EN]
Style: {style_label}
Seed: {seed}
Duration: {duration_sec}s @ {fps}fps
Resolution: {resolution}
Audio: {"present" if has_audio else "none"} — {audio_line_en}
Technique: pure DSP (OpenCV/NumPy/SciPy), no neural networks.
Aesthetic references: anatomical plastination (von Hagens), clinical
hyperrealist sculpture (Mueck, Jinks) — reinterpreted as an algorithmic process.
""".strip()
    return report


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.caption(
        "Scomposizione anatomica algoritmica da una singola foto. "
        "Algorithmic anatomical decomposition from a single photo."
    )

    if "output_path" not in st.session_state:
        st.session_state.output_path = None
        st.session_state.report_text = None

    image_file = st.file_uploader(
        "Foto / Photo (jpg, png)", type=["jpg", "jpeg", "png"], key="uploader_image",
    )
    audio_file = st.file_uploader(
        "Audio opzionale / Optional audio (mp3, wav) — guida il ritmo della deformazione",
        type=["mp3", "wav"], key="uploader_audio",
    )

    style_options = list(STYLE_LABELS.keys())
    style_key = st.selectbox(
        "Stile / Style",
        options=style_options,
        format_func=lambda k: STYLE_LABELS[k],
        index=0,
        key="select_style",
    )

    st.subheader("Formato / Format")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        aspect_label = st.selectbox(
            "Aspect ratio", options=list(ASPECT_PRESETS.keys()), index=0,
            key="select_aspect",
        )
    with col_res2:
        quick_preview = st.checkbox(
            "Render veloce, mezza risoluzione / Fast render, half-res",
            value=False,
            key="check_quick_preview",
            help="Dimezza la risoluzione del FILE FINALE (non solo dell'anteprima) "
                 "per velocizzare i test. Lascia disattivato per scaricare alla "
                 "risoluzione selezionata sopra. / Halves the FINAL FILE resolution "
                 "(not just the preview) to speed up testing. Leave unchecked to "
                 "download at the resolution selected above.",
        )

    target_w, target_h = ASPECT_PRESETS[aspect_label]
    if quick_preview:
        target_w, target_h = target_w // 2, target_h // 2
    st.caption(f"Risoluzione di render / Render resolution: {target_w}x{target_h}")

    st.subheader("Durata / Duration")

    # Struttura DOM stabile: il checkbox esiste sempre (disabilitato senza audio),
    # cosi' l'albero dei widget non cambia tra i rerun e si evita il conflitto
    # di riconciliazione React ("removeChild") osservato su Streamlit Cloud.
    audio_duration = None
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_probe:
            tmp_probe.write(audio_file.getvalue())
            probe_path = tmp_probe.name
        try:
            audio_duration = get_audio_duration(probe_path)
        except Exception:
            audio_duration = None
        finally:
            os.unlink(probe_path)

    duration_checkbox_label = (
        f"Usa la durata del brano ({audio_duration:.1f}s) / Use track duration"
        if audio_duration is not None
        else "Usa la durata del brano / Use track duration (carica un audio)"
    )
    use_audio_duration = st.checkbox(
        duration_checkbox_label,
        value=True,
        disabled=(audio_duration is None),
        key="check_use_audio_duration",
    )

    if use_audio_duration and audio_duration is not None:
        duration_sec = min(audio_duration, MAX_DURATION_SEC)
        if audio_duration > MAX_DURATION_SEC:
            st.warning(
                f"Brano piu' lungo di {MAX_DURATION_SEC}s: video troncato a "
                f"{MAX_DURATION_SEC}s. / Track longer than {MAX_DURATION_SEC}s: "
                f"video capped at {MAX_DURATION_SEC}s."
            )
    else:
        duration_sec = st.slider(
            "Durata (s) / Duration (s)", 3, MAX_DURATION_SEC, 15,
            key="slider_duration",
            help="Video lunghi = tempo di rendering molto maggiore. "
                 "Long videos = much longer render time.",
        )

    if duration_sec > 60:
        st.info(
            "Durate superiori al minuto possono richiedere diversi minuti di "
            "rendering, specialmente a piena risoluzione. / Durations over a "
            "minute may take several minutes to render, especially at full resolution."
        )

    col1, col2 = st.columns(2)
    with col1:
        seed = st.number_input("Seed", min_value=0, value=7, step=1, key="input_seed")
    with col2:
        max_strength = st.slider(
            "Intensita' massima / Max intensity", 10, 100, 55, key="slider_strength",
        )

    fps = 24

    render_clicked = st.button("Genera / Render", type="primary", key="button_render")

    if render_clicked:
        if image_file is None:
            st.error("Carica una foto prima di procedere. / Upload a photo first.")
            return

        if style_key not in IMPLEMENTED_STYLES:
            st.error(
                f"Lo stile '{STYLE_LABELS[style_key]}' non e' ancora implementato. "
                f"Style '{STYLE_LABELS[style_key]}' is not implemented yet."
            )
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.jpg")
            with open(img_path, "wb") as fh:
                fh.write(image_file.getvalue())

            audio_path = None
            if audio_file is not None:
                audio_path = os.path.join(tmpdir, "input_audio.mp3")
                with open(audio_path, "wb") as fh:
                    fh.write(audio_file.getvalue())

            progress = st.progress(0, text="Analisi audio / Audio analysis...")

            bpm_value = None
            if audio_path is not None and LIBROSA_OK:
                envelope, beat_frames, bpm_value = analyze_audio(audio_path, fps, duration_sec)
            else:
                if audio_path is not None and not LIBROSA_OK:
                    st.warning("librosa non disponibile: uso envelope sintetico. / "
                               "librosa unavailable: using synthetic envelope.")
                envelope, beat_frames, bpm_value = synthetic_envelope(fps, duration_sec)

            progress.progress(15, text="Caricamento immagine / Loading image...")
            base_img = load_image_fit_aspect(img_path, target_w, target_h)
            subject_mask = build_subject_mask(base_img)

            progress.progress(30, text="Rendering frame / Rendering frames...")
            t0 = time.time()

            if STYLE_RENDERERS[style_key] == "melt":
                frames = render_melt(
                    base_img, subject_mask, envelope, beat_frames, int(seed),
                    max_strength=float(max_strength), max_layers=3, layer_offset=7,
                )
            elif STYLE_RENDERERS[style_key] == "voronoi":
                frames = render_voronoi(
                    base_img, envelope, beat_frames, int(seed),
                    max_intensity=float(max_strength) / 40.0, n_points=28,
                )
            elif STYLE_RENDERERS[style_key] == "reaction_diffusion":
                frames = render_reaction_diffusion(
                    base_img, subject_mask, envelope, beat_frames, int(seed),
                    max_intensity=float(max_strength) / 60.0,
                )
            elif STYLE_RENDERERS[style_key] == "depth_peel":
                frames = render_depth_peel(
                    base_img, subject_mask, envelope, beat_frames, int(seed),
                    max_intensity=float(max_strength) / 40.0,
                )
            elif STYLE_RENDERERS[style_key] == "capillary_bleed":
                frames = render_capillary_bleed(
                    base_img, subject_mask, envelope, beat_frames, int(seed),
                    max_intensity=float(max_strength) / 60.0, n_walkers=40,
                )
            elif STYLE_RENDERERS[style_key] == "fluid_melt":
                frames = render_fluid_melt(
                    base_img, subject_mask, envelope, beat_frames, int(seed),
                    max_intensity=float(max_strength) * 0.9,
                )
            else:
                st.error("Stile non riconosciuto. / Unrecognized style.")
                return

            elapsed = time.time() - t0
            progress.progress(70, text=f"Frame renderizzati in {elapsed:.1f}s / Encoding...")

            raw_video_path = os.path.join(tmpdir, "raw.mp4")
            write_raw_video(frames, fps, raw_video_path)

            progress.progress(85, text="Transcodifica H.264 + mux audio / "
                                        "H.264 transcode + audio mux...")
            final_path = os.path.join(tmpdir, "bodyerror_output.mp4")
            try:
                finalize_video(raw_video_path, audio_path, duration_sec, final_path)
            except subprocess.CalledProcessError as exc:
                st.error(f"Errore ffmpeg / ffmpeg error: {exc.stderr.decode(errors='ignore')[-500:]}")
                return

            output_bytes_path = os.path.join(tempfile.gettempdir(),
                                              f"bodyerror_{int(time.time())}.mp4")
            with open(final_path, "rb") as src, open(output_bytes_path, "wb") as dst:
                dst.write(src.read())

            st.session_state.output_path = output_bytes_path
            st.session_state.report_text = build_report(
                style_key, int(seed), duration_sec, fps,
                f"{target_w}x{target_h} ({aspect_label})",
                audio_path is not None, bpm_value,
            )

            progress.progress(100, text="Completato / Done")

    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        st.video(st.session_state.output_path)
        with open(st.session_state.output_path, "rb") as fh:
            st.download_button(
                "Scarica video / Download video",
                data=fh.read(),
                file_name="bodyerror_output.mp4",
                mime="video/mp4",
                key="button_download",
            )
        st.text_area("Report", st.session_state.report_text, height=220, key="area_report")


if __name__ == "__main__":
    main()
