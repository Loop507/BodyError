# -*- coding: utf-8 -*-
"""
BodyError // Loop507
=====================
Trasforma una foto in un video "Body Error Realistico": scomposizione
anatomica REALE, ancorata ai landmark del volto (dlib, 68 punti), guidata
da 3 bande audio separate (bassi/medi/alti) con controlli indipendenti.
Pure DSP / algoritmico - nessuna rete neurale generativa (dlib usa un
modello di regressione classico per i landmark, non genera contenuto).

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
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import dlib
    import face_recognition_models
    DLIB_OK = True
except ImportError:
    DLIB_OK = False


# ---------------------------------------------------------------------------
# COSTANTI
# ---------------------------------------------------------------------------

APP_TITLE = "BodyError // Loop507"

STYLE_ANATOMICAL = "anatomical_warp"
STYLE_VORONOI = "voronoi_fracture"
STYLE_CAPILLARY = "capillary_bleed"

STYLE_LABELS = {
    STYLE_ANATOMICAL: "Anatomical Warp",
    STYLE_VORONOI: "Voronoi Fracture",
    STYLE_CAPILLARY: "Capillary Bleed",
}

ASPECT_PRESETS = {
    "16:9  (1280x720)": (1280, 720),
    "9:16  (720x1280)": (720, 1280),
    "1:1   (720x720)": (720, 720),
}

MAX_DURATION_SEC = 300  # 5 minuti, con avviso sul tempo di rendering

# gruppi anatomici standard dlib 68-punti
LANDMARK_GROUPS = {
    "jaw": list(range(0, 17)),
    "eyebrow_r": list(range(17, 22)),
    "eyebrow_l": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "eye_r": list(range(36, 42)),
    "eye_l": list(range(42, 48)),
    "mouth": list(range(48, 68)),
}


# ---------------------------------------------------------------------------
# UTILITY IMMAGINE (invariate, gia' testate)
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
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]

    interp = cv2.INTER_AREA if img.shape[0] > target_h else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (target_w, target_h), interpolation=interp)
    return img.astype(np.float32) / 255.0


def build_background_subject_mask(img):
    """Fallback euristico (nessun volto rilevato): distingue soggetto da sfondo uniforme."""
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
# LANDMARK DEL VOLTO (dlib, 68 punti)
# ---------------------------------------------------------------------------

_DLIB_DETECTOR = None
_DLIB_PREDICTOR = None


def _get_dlib_models():
    global _DLIB_DETECTOR, _DLIB_PREDICTOR
    if _DLIB_DETECTOR is None:
        _DLIB_DETECTOR = dlib.get_frontal_face_detector()
        _DLIB_PREDICTOR = dlib.shape_predictor(
            face_recognition_models.pose_predictor_model_location())
    return _DLIB_DETECTOR, _DLIB_PREDICTOR


def detect_landmarks(img_float_bgr):
    """Restituisce (68,2) landmark in coordinate pixel, o None se nessun volto."""
    if not DLIB_OK:
        return None
    detector, predictor = _get_dlib_models()
    img_u8 = (np.clip(img_float_bgr, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return pts



# ---------------------------------------------------------------------------
# ANALISI AUDIO A 3 BANDE (bassi / medi / alti)
# ---------------------------------------------------------------------------

def get_audio_duration(path):
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def _normalize_envelope(energy, energy_times, video_times):
    env = np.interp(video_times, energy_times, energy)
    p5, p95 = np.percentile(env, [5, 95])
    env = np.clip((env - p5) / max(p95 - p5, 1e-6), 0, 1)
    kernel = np.ones(5) / 5
    env = np.convolve(env, kernel, mode="same")
    return env


def analyze_audio_bands(path, target_fps, duration_sec):
    """
    Estrae 3 envelope di energia indipendenti (bassi/medi/alti) via STFT,
    piu' i frame dei beat e il BPM stimato, per pilotare separatamente le
    diverse componenti anatomiche della deformazione.
    """
    y, sr = librosa.load(path, sr=22050, mono=True, duration=duration_sec)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    bass_idx = np.where(freqs <= 250)[0]
    mid_idx = np.where((freqs > 250) & (freqs <= 2000))[0]
    high_idx = np.where(freqs > 2000)[0]

    energy_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=512)

    tempo, beat_frames_idx = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames_idx, sr=sr, hop_length=512)

    total_frames = int(duration_sec * target_fps)
    video_times = np.linspace(0, duration_sec, total_frames)

    env_bass = _normalize_envelope(stft[bass_idx].mean(axis=0), energy_times, video_times)
    env_mid = _normalize_envelope(stft[mid_idx].mean(axis=0), energy_times, video_times)
    env_high = _normalize_envelope(stft[high_idx].mean(axis=0), energy_times, video_times)

    beat_video_frames = sorted(set(
        int(np.clip(bt / duration_sec * total_frames, 0, total_frames - 1))
        for bt in beat_times if bt <= duration_sec
    ))

    bpm_value = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return env_bass, env_mid, env_high, beat_video_frames, bpm_value


def synthetic_bands(target_fps, duration_sec):
    """
    Fallback se non c'e' audio: tre curve ADSR-style leggermente sfasate,
    cosi' anche senza musica le tre componenti (bassi/medi/alti) non sono
    identiche tra loro.
    """
    total_frames = int(duration_sec * target_fps)
    t = np.linspace(0, 1, total_frames)

    def adsr(phase_offset):
        tp = np.clip(t + phase_offset, 0, 1)
        attack_end, decay_end, sustain_end = 0.25, 0.4, 0.85
        env = np.zeros(total_frames)
        for i, tv in enumerate(tp):
            if tv < attack_end:
                env[i] = tv / attack_end
            elif tv < decay_end:
                local = (tv - attack_end) / (decay_end - attack_end)
                env[i] = 1.0 - 0.25 * local
            elif tv < sustain_end:
                local = (tv - decay_end) / (sustain_end - decay_end)
                env[i] = 0.75 + 0.1 * np.sin(local * np.pi * 3) * 0.5 + 0.1 * local
            else:
                local = (tv - sustain_end) / max(1 - sustain_end, 1e-6)
                env[i] = 0.9 + 0.1 * local
        return np.clip(env, 0, 1)

    env_bass = adsr(0.0)
    env_mid = adsr(0.03)
    env_high = adsr(-0.03)
    beat_frames = list(range(0, total_frames, max(int(target_fps * 0.5), 1)))
    return env_bass, env_mid, env_high, beat_frames, None


# ---------------------------------------------------------------------------
# STILE: ANATOMICAL WARP (motore principale)
# ---------------------------------------------------------------------------

def build_dynamic_displacement(pts, jaw_i, mouth_i, eye_i, rng):
    """
    Sposta i landmark con logica anatomica specifica per gruppo, con
    intensita' indipendenti per gruppo (guidate dalle bande audio a monte).
    Ricalcolato ogni frame a partire dai landmark ORIGINALI: la geometria
    reagisce in tempo reale all'energia della musica, non solo cresce.
    """
    displaced = pts.copy()

    eye_r_center = pts[LANDMARK_GROUPS["eye_r"]].mean(axis=0)
    for i in LANDMARK_GROUPS["eye_r"]:
        displaced[i] = pts[i] + (pts[i] - eye_r_center) * (0.9 * eye_i)

    eye_l_center = pts[LANDMARK_GROUPS["eye_l"]].mean(axis=0)
    for i in LANDMARK_GROUPS["eye_l"]:
        displaced[i] = pts[i] - (pts[i] - eye_l_center) * (0.35 * eye_i)

    mouth_center = pts[LANDMARK_GROUPS["mouth"]].mean(axis=0)
    for i in LANDMARK_GROUPS["mouth"]:
        dx = (pts[i][0] - mouth_center[0]) * (1.1 * mouth_i)
        dy = 16.0 * mouth_i
        displaced[i] = pts[i] + np.array([dx, dy])

    jaw_center_x = pts[LANDMARK_GROUPS["jaw"]][:, 0].mean()
    jaw_span = np.ptp(pts[LANDMARK_GROUPS["jaw"]][:, 0])
    for i in LANDMARK_GROUPS["jaw"]:
        dist_ratio = abs(pts[i][0] - jaw_center_x) / max(jaw_span / 2, 1.0)
        dy = (32.0 + 58.0 * dist_ratio) * jaw_i
        displaced[i] = pts[i] + np.array([rng.uniform(-3, 3), dy])

    for i in LANDMARK_GROUPS["nose"]:
        dx = rng.uniform(-3, 3) * (mouth_i * 0.3)
        displaced[i] = pts[i] + np.array([dx, 2.0 * mouth_i * 0.3])

    return displaced


def compute_displacement_field(src_pts, dst_pts, shape, eval_scale=0.2):
    """
    Interpola lo spostamento landmark->landmark su un campo denso (via RBF
    thin-plate-spline), calcolato a bassa risoluzione per velocita' e poi
    ricampionato alla risoluzione piena. I bordi immagine sono ancorati a
    spostamento zero cosi' il warp resta confinato al volto.
    """
    h, w = shape[:2]
    eh, ew = max(int(h * eval_scale), 40), max(int(w * eval_scale), 40)
    sx, sy = w / ew, h / eh

    pts_s = src_pts / np.array([sx, sy])
    disp_s = dst_pts / np.array([sx, sy])

    border = np.array([
        [-0.2 * ew, -0.2 * eh], [ew * 0.5, -0.2 * eh], [ew * 1.2, -0.2 * eh],
        [-0.2 * ew, eh * 0.5], [ew * 1.2, eh * 0.5],
        [-0.2 * ew, eh * 1.2], [ew * 0.5, eh * 1.2], [ew * 1.2, eh * 1.2],
    ], dtype=np.float32)

    src_full = np.concatenate([pts_s, border], axis=0)
    dst_full = np.concatenate([disp_s, border], axis=0)
    displacement = dst_full - src_full

    rbf_x = RBFInterpolator(src_full, displacement[:, 0], kernel="thin_plate_spline")
    rbf_y = RBFInterpolator(src_full, displacement[:, 1], kernel="thin_plate_spline")

    xx, yy = np.meshgrid(np.arange(ew), np.arange(eh))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    dx_small = rbf_x(grid).reshape(eh, ew).astype(np.float32)
    dy_small = rbf_y(grid).reshape(eh, ew).astype(np.float32)

    dx_full = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC) * sx
    dy_full = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC) * sy
    return dx_full, dy_full


def render_anatomical_warp(base_img, pts, env_bass, env_mid, env_high, beat_frames,
                            seed, base_intensity, w_bass, w_mid, w_high, growth_rate):
    """
    Motore principale: deformazione REALE ancorata ai 68 landmark del volto.
    - una componente permanente e crescente (danno progressivo, velocita'
      legata a growth_rate e all'energia media della musica)
    - una componente istantanea proporzionale alle 3 bande audio, ognuna
      pesata dai controlli utente (w_bass -> mascella, w_mid -> bocca,
      w_high -> occhi)
    """
    rng = np.random.default_rng(seed)
    h, w = base_img.shape[:2]
    xx_base, yy_base = np.meshgrid(np.arange(w).astype(np.float32),
                                    np.arange(h).astype(np.float32))

    total_frames = len(env_bass)
    growth_acc = 0.0
    frames = []

    for f in range(total_frames):
        eb, em, eh_ = float(env_bass[f]), float(env_mid[f]), float(env_high[f])
        avg_e = (eb + em + eh_) / 3.0
        growth_acc = min(1.0, growth_acc + (avg_e / total_frames) * growth_rate)

        permanent = growth_acc * base_intensity * 0.6
        jaw_i = permanent + eb * w_bass * base_intensity
        mouth_i = permanent * 0.6 + em * w_mid * base_intensity
        eye_i = permanent * 0.4 + eh_ * w_high * base_intensity

        if f in beat_frames:
            jaw_i *= 1.4

        displaced = build_dynamic_displacement(pts, jaw_i, mouth_i, eye_i, rng)
        dx, dy = compute_displacement_field(pts, displaced, base_img.shape)

        new_x = np.clip(xx_base + dx, 0, w - 1)
        new_y = np.clip(yy_base + dy, 0, h - 1)
        frame = cv2.remap(base_img, new_x, new_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# STILE: VORONOI FRACTURE (rifatto: niente linee finte, contenuto reale
# nei varchi, separazione guidata dai bassi)
# ---------------------------------------------------------------------------

def seed_points_in_region(gray, region_mask, n_points, rng):
    edges = cv2.Canny(gray, 50, 130).astype(np.float32) * region_mask
    ys, xs = np.where(edges > 0)
    h, w = gray.shape
    if len(xs) < n_points:
        extra = n_points - len(xs)
        ys_r, xs_r = np.where(region_mask > 0.3)
        if len(xs_r) == 0:
            xs_r = rng.integers(0, w, extra)
            ys_r = rng.integers(0, h, extra)
        idx_r = rng.choice(len(xs_r), min(extra, len(xs_r)), replace=len(xs_r) < extra)
        xs = np.concatenate([xs, xs_r[idx_r]])
        ys = np.concatenate([ys, ys_r[idx_r]])
    idx = rng.choice(len(xs), min(n_points, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def build_cell_labels(shape, points):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(points)
    _, labels = tree.query(coords, k=1)
    return labels.reshape(h, w)


def voronoi_fracture_frame(img, region_mask, intensity, n_points, rng):
    h, w = img.shape[:2]
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    points = seed_points_in_region(gray_blur, region_mask, n_points, rng)
    labels = build_cell_labels((h, w), points)
    n_cells = len(points)
    cx = points[:, 0].mean()
    cy = points[:, 1].mean()

    disp_x = np.zeros(n_cells, dtype=np.float32)
    disp_y = np.zeros(n_cells, dtype=np.float32)
    for i, (px, py) in enumerate(points):
        dx, dy = px - cx, py - cy
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        push = (dist / max(w, h)) * 1.2
        disp_x[i] = (dx / dist) * push * intensity * 45
        disp_y[i] = (dy / dist) * push * intensity * 45 + 0.7 * intensity * 30

    disp_x += rng.uniform(-6, 6, n_cells) * intensity
    disp_y += rng.uniform(-3, 10, n_cells) * intensity

    # base "tessuto sotto la crepa": la foto stessa leggermente ammorbidita,
    # NON annerita - cosi' dove una placca si sposta si vede ancora il volto
    # sottostante (materia reale), mai un vuoto/crepa nera
    underlayer = cv2.GaussianBlur(img, (9, 9), 0) * 0.92

    result = underlayer.copy()

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

    return np.clip(result, 0, 1)


def render_voronoi(base_img, region_mask, env_bass, env_mid, env_high, beat_frames,
                    seed, base_intensity, n_points, growth_rate):
    rng_seed_stream = np.random.default_rng(seed)
    total_frames = len(env_bass)
    frames = []
    growth_acc = 0.0
    seed_offset = 0

    for f in range(total_frames):
        eb, em, eh_ = float(env_bass[f]), float(env_mid[f]), float(env_high[f])
        avg_e = (eb + em + eh_) / 3.0
        growth_acc = min(1.0, growth_acc + (avg_e / total_frames) * growth_rate)

        if f in beat_frames:
            seed_offset += 1
        local_rng = np.random.default_rng(seed + seed_offset)

        intensity = 0.1 + base_intensity * (growth_acc * 0.5 + eb * 0.5)
        frame = voronoi_fracture_frame(base_img, region_mask, intensity, n_points, local_rng)
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    _ = rng_seed_stream
    return frames


# ---------------------------------------------------------------------------
# STILE: CAPILLARY BLEED (rifatto: ancorato ai landmark, reattivo a bande)
# ---------------------------------------------------------------------------

def render_capillary_bleed(base_img, region_mask, pts, env_bass, env_mid, env_high,
                            beat_frames, seed, base_intensity, n_walkers=40):
    rng = np.random.default_rng(seed)
    h, w = base_img.shape[:2]

    gray = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    gx_n = gx / mag
    gy_n = gy / mag

    # semi di partenza: i landmark stessi (contorni anatomici reali), se
    # disponibili, altrimenti bordi ad alto contrasto dentro la region_mask
    if pts is not None:
        seed_idx = (LANDMARK_GROUPS["eye_r"] + LANDMARK_GROUPS["eye_l"]
                    + LANDMARK_GROUPS["mouth"] + LANDMARK_GROUPS["jaw"]
                    + LANDMARK_GROUPS["nose"])
        candidates = pts[seed_idx]
        idx = rng.choice(len(candidates), min(n_walkers, len(candidates)), replace=False)
        walker_pos = candidates[idx].copy().astype(np.float32)
    else:
        edges = cv2.Canny(gray_blur, 50, 130).astype(np.float32) * region_mask
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            ys, xs = np.array([h // 2]), np.array([w // 2])
        idx = rng.choice(len(xs), min(n_walkers, len(xs)), replace=False)
        walker_pos = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    walker_dir = rng.uniform(-1, 1, walker_pos.shape).astype(np.float32)
    vein_mask = np.zeros((h, w), dtype=np.float32)
    bleed_color = np.array([0.05, 0.02, 0.35], dtype=np.float32)  # BGR rosso scuro

    frames = []
    total_frames = len(env_bass)
    max_walkers = n_walkers * 5

    for f in range(total_frames):
        eb, em, eh_ = float(env_bass[f]), float(env_mid[f]), float(env_high[f])

        step_size = 1.0 + 3.0 * em  # i medi guidano la velocita' di crescita

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

        # gli ALTI pilotano la frequenza di ramificazione (nuove diramazioni)
        branch_prob = eh_ * 0.5
        if (f in beat_frames or rng.uniform(0, 1) < branch_prob) and len(walker_pos) < max_walkers:
            n_new = min(4, n_walkers)
            branch_idx = rng.integers(0, len(walker_pos), n_new)
            new_pos = walker_pos[branch_idx] + rng.uniform(-2, 2, (n_new, 2))
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0, w - 1)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0, h - 1)
            new_dir = rng.uniform(-1, 1, (n_new, 2)).astype(np.float32)
            walker_pos = np.concatenate([walker_pos, new_pos.astype(np.float32)], axis=0)
            walker_dir = np.concatenate([walker_dir, new_dir], axis=0)

        vein_blur = cv2.GaussianBlur(vein_mask, (3, 3), 0)
        vein_blur = np.clip(vein_blur * region_mask, 0, 1)

        # i BASSI pilotano lo spessore/opacita' delle venature
        alpha = vein_blur[..., None] * (0.4 + 0.5 * eb) * base_intensity
        frame = base_img * (1 - alpha) + bleed_color * alpha
        frame = clinical_grade(frame)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# EXPORT VIDEO (invariato, gia' testato)
# ---------------------------------------------------------------------------

def write_raw_video(frames, fps, out_path):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()


def finalize_video(raw_video_path, audio_path, duration_sec, out_path):
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

def build_report(style_key, seed, duration_sec, fps, resolution, has_audio, bpm_value,
                  weights_used):
    style_label = STYLE_LABELS[style_key]
    audio_line_it = f"BPM rilevato: {bpm_value:.1f}" if bpm_value else "Nessun audio: envelope sintetico"
    audio_line_en = f"Detected BPM: {bpm_value:.1f}" if bpm_value else "No audio: synthetic envelope"

    report = f"""
BodyError // Loop507 :: REPORT

[IT]
Stile: {style_label}
Seed: {seed}
Durata: {duration_sec:.1f}s @ {fps}fps
Risoluzione: {resolution}
Audio: {"presente" if has_audio else "assente"} — {audio_line_it}
Pesi banda (bassi/medi/alti): {weights_used}
Tecnica: landmark del volto (dlib, 68 punti) + DSP puro (OpenCV/NumPy/SciPy),
nessuna rete neurale generativa.
Riferimenti estetici: plastinazione anatomica (von Hagens), iperrealismo
scultoreo clinico (Mueck, Jinks) — reinterpretati come processo algoritmico.

[EN]
Style: {style_label}
Seed: {seed}
Duration: {duration_sec:.1f}s @ {fps}fps
Resolution: {resolution}
Audio: {"present" if has_audio else "none"} — {audio_line_en}
Band weights (bass/mid/high): {weights_used}
Technique: face landmarks (dlib, 68 points) + pure DSP (OpenCV/NumPy/SciPy),
no generative neural network.
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
        "Deformazione anatomica reale ancorata ai landmark del volto, "
        "guidata da 3 bande audio indipendenti. / Real anatomical "
        "deformation anchored to face landmarks, driven by 3 independent "
        "audio bands."
    )

    if not DLIB_OK:
        st.error(
            "dlib non disponibile in questo ambiente: gli stili basati sui "
            "landmark del volto non funzioneranno. / dlib not available in "
            "this environment: face-landmark-based styles will not work."
        )

    if "output_path" not in st.session_state:
        st.session_state.output_path = None
        st.session_state.report_text = None

    image_file = st.file_uploader(
        "Foto / Photo (jpg, png) — un volto ben visibile / a clearly visible face",
        type=["jpg", "jpeg", "png"], key="uploader_image",
    )
    audio_file = st.file_uploader(
        "Audio opzionale / Optional audio (mp3, wav) — guida la deformazione",
        type=["mp3", "wav"], key="uploader_audio",
    )

    style_options = list(STYLE_LABELS.keys())
    style_key = st.selectbox(
        "Stile / Style", options=style_options,
        format_func=lambda k: STYLE_LABELS[k], index=0, key="select_style",
    )

    st.subheader("Reattivita' audio / Audio reactivity")
    st.caption(
        "Quanto ogni banda di frequenza guida la deformazione. / How much "
        "each frequency band drives the deformation."
    )
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        w_bass = st.slider("Bassi / Bass", 0.0, 2.0, 1.0, 0.1, key="slider_bass")
    with col_b2:
        w_mid = st.slider("Medi / Mid", 0.0, 2.0, 1.0, 0.1, key="slider_mid")
    with col_b3:
        w_high = st.slider("Alti / High", 0.0, 2.0, 1.0, 0.1, key="slider_high")

    st.subheader("Controlli per stile / Per-style controls")
    with st.expander("Anatomical Warp", expanded=(style_key == STYLE_ANATOMICAL)):
        aw_intensity = st.slider("Intensita' / Intensity", 0.1, 3.0, 1.2, 0.1,
                                  key="aw_intensity")
        aw_growth = st.slider("Velocita' progressione permanente / Permanent growth rate",
                               0.5, 5.0, 2.0, 0.5, key="aw_growth")
    with st.expander("Voronoi Fracture", expanded=(style_key == STYLE_VORONOI)):
        vf_intensity = st.slider("Intensita' frattura / Fracture intensity", 0.2, 3.0, 1.0,
                                  0.1, key="vf_intensity")
        vf_points = st.slider("Numero placche / Number of pieces", 10, 60, 26, 2,
                               key="vf_points")
        vf_growth = st.slider("Velocita' progressione / Growth rate", 0.5, 5.0, 2.0, 0.5,
                               key="vf_growth")
    with st.expander("Capillary Bleed", expanded=(style_key == STYLE_CAPILLARY)):
        cb_intensity = st.slider("Intensita' venature / Vein intensity", 0.2, 3.0, 1.0, 0.1,
                                  key="cb_intensity")
        cb_walkers = st.slider("Numero venature iniziali / Initial vein count", 10, 80, 40,
                                5, key="cb_walkers")

    st.subheader("Formato / Format")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        aspect_label = st.selectbox("Aspect ratio", options=list(ASPECT_PRESETS.keys()),
                                     index=0, key="select_aspect")
    with col_res2:
        quick_preview = st.checkbox(
            "Render veloce, mezza risoluzione / Fast render, half-res",
            value=False, key="check_quick_preview",
            help="Dimezza la risoluzione del FILE FINALE per velocizzare i test. "
                 "/ Halves the FINAL FILE resolution to speed up testing.",
        )

    target_w, target_h = ASPECT_PRESETS[aspect_label]
    if quick_preview:
        target_w, target_h = target_w // 2, target_h // 2
    st.caption(f"Risoluzione di render / Render resolution: {target_w}x{target_h}")

    st.subheader("Durata / Duration")
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
        duration_checkbox_label, value=True, disabled=(audio_duration is None),
        key="check_use_audio_duration",
    )

    if use_audio_duration and audio_duration is not None:
        duration_sec = min(audio_duration, MAX_DURATION_SEC)
        if audio_duration > MAX_DURATION_SEC:
            st.warning(
                f"Brano piu' lungo di {MAX_DURATION_SEC}s: video troncato. / "
                f"Track longer than {MAX_DURATION_SEC}s: video capped."
            )
    else:
        duration_sec = st.slider(
            "Durata (s) / Duration (s)", 3, MAX_DURATION_SEC, 15, key="slider_duration",
            help="Video lunghi = tempo di rendering molto maggiore. / "
                 "Long videos = much longer render time.",
        )

    if duration_sec > 60:
        st.info(
            "Durate superiori al minuto possono richiedere diversi minuti di "
            "rendering. / Durations over a minute may take several minutes to render."
        )

    seed = st.number_input("Seed", min_value=0, value=7, step=1, key="input_seed")
    fps = 24

    render_clicked = st.button("Genera / Render", type="primary", key="button_render")

    if render_clicked:
        if image_file is None:
            st.error("Carica una foto prima di procedere. / Upload a photo first.")
            return
        if not DLIB_OK:
            st.error("dlib non disponibile: impossibile procedere. / "
                      "dlib not available: cannot proceed.")
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
                env_bass, env_mid, env_high, beat_frames, bpm_value = analyze_audio_bands(
                    audio_path, fps, duration_sec)
            else:
                if audio_path is not None and not LIBROSA_OK:
                    st.warning("librosa non disponibile: uso envelope sintetico. / "
                               "librosa unavailable: using synthetic envelope.")
                env_bass, env_mid, env_high, beat_frames, bpm_value = synthetic_bands(
                    fps, duration_sec)

            progress.progress(15, text="Caricamento immagine / Loading image...")
            base_img = load_image_fit_aspect(img_path, target_w, target_h)

            progress.progress(25, text="Rilevamento volto / Face detection...")
            pts = detect_landmarks(base_img)
            if pts is None:
                st.warning(
                    "Nessun volto rilevato: gli stili basati sui landmark saranno "
                    "limitati. / No face detected: landmark-based styles will be "
                    "limited."
                )

            # Voronoi e Capillary Bleed devono coprire l'intero soggetto (utile
            # anche per foto a figura intera), non solo il piccolo poligono del
            # volto - quindi usano sempre la sagoma sfondo/primo piano.
            region_mask = build_background_subject_mask(base_img)

            if style_key == STYLE_ANATOMICAL and pts is None:
                st.error(
                    "Anatomical Warp richiede un volto rilevabile nella foto: "
                    "deforma solo i tratti del viso (occhi/naso/bocca/mascella), "
                    "non il corpo intero. Per una figura intera prova Voronoi "
                    "Fracture o Capillary Bleed. / Anatomical Warp requires a "
                    "detectable face: it only deforms facial features, not the "
                    "whole body. For a full-body photo try Voronoi Fracture or "
                    "Capillary Bleed instead."
                )
                return

            progress.progress(35, text="Rendering frame / Rendering frames...")
            t0 = time.time()

            if style_key == STYLE_ANATOMICAL:
                frames = render_anatomical_warp(
                    base_img, pts, env_bass, env_mid, env_high, beat_frames, int(seed),
                    base_intensity=float(aw_intensity), w_bass=float(w_bass),
                    w_mid=float(w_mid), w_high=float(w_high), growth_rate=float(aw_growth),
                )
            elif style_key == STYLE_VORONOI:
                frames = render_voronoi(
                    base_img, region_mask, env_bass, env_mid, env_high, beat_frames,
                    int(seed), base_intensity=float(vf_intensity) * (0.5 + w_bass * 0.5),
                    n_points=int(vf_points), growth_rate=float(vf_growth),
                )
            elif style_key == STYLE_CAPILLARY:
                frames = render_capillary_bleed(
                    base_img, region_mask, pts, env_bass, env_mid, env_high, beat_frames,
                    int(seed), base_intensity=float(cb_intensity), n_walkers=int(cb_walkers),
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
                f"{w_bass:.1f} / {w_mid:.1f} / {w_high:.1f}",
            )

            progress.progress(100, text="Completato / Done")

    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        st.video(st.session_state.output_path)
        with open(st.session_state.output_path, "rb") as fh:
            st.download_button(
                "Scarica video / Download video", data=fh.read(),
                file_name="bodyerror_output.mp4", mime="video/mp4", key="button_download",
            )
        st.text_area("Report", st.session_state.report_text, height=240, key="area_report")


if __name__ == "__main__":
    main()
