# -*- coding: utf-8 -*-
"""BodyError // Loop507 - foto -> video di scomposizione anatomica, ancorata
ai landmark del volto (dlib) e guidata da 3 bande audio (bassi/medi/alti)."""

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import dlib
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
STYLE_COMBO = "voronoi_capillary_combo"

STYLE_LABELS = {
    STYLE_ANATOMICAL: "Anatomical Warp",
    STYLE_VORONOI: "Voronoi Fracture",
    STYLE_CAPILLARY: "Capillary Bleed",
    STYLE_COMBO: "Voronoi + Capillary (combo)",
}

ASPECT_PRESETS = {
    "16:9  (1280x720)": (1280, 720),
    "9:16  (720x1280)": (720, 1280),
    "1:1   (720x720)": (720, 720),
}

MAX_DURATION_SEC = 300  # 5 minuti, con avviso sul tempo di rendering

# Modello dlib scaricato a runtime da un mirror GitHub (raw.githubusercontent.com)
# invece del pacchetto pip "face_recognition_models" (100MB, solo sorgente,
# spesso in timeout su piattaforme con risorse limitate come Streamlit Cloud).
DLIB_MODEL_URL = (
    "https://raw.githubusercontent.com/davisking/dlib-models/master/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)
DLIB_MODEL_PATH = os.path.join(tempfile.gettempdir(), "shape_predictor_68_face_landmarks.dat")

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

def compute_fit_aspect_crop(src_w, src_h, target_w, target_h):
    """Calcola il rettangolo di center-crop (x0, y0, w, h) che porta l'immagine
    sorgente all'aspect ratio target, senza deformarla."""
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        crop_w = int(src_h * target_ratio)
        x0 = (src_w - crop_w) // 2
        return x0, 0, crop_w, src_h
    else:
        crop_h = int(src_w / target_ratio)
        y0 = (src_h - crop_h) // 2
        return 0, y0, src_w, crop_h


def load_image_fit_aspect(path, target_w, target_h):
    """Carica l'immagine e la adatta alla risoluzione target (center-crop + resize)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Immagine non leggibile / Image could not be read")

    h, w = img.shape[:2]
    x0, y0, crop_w, crop_h = compute_fit_aspect_crop(w, h, target_w, target_h)
    img = img[y0:y0 + crop_h, x0:x0 + crop_w]

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


def _ensure_dlib_model():
    """Scarica e decomprime il modello dlib al primo utilizzo, se non gia' presente."""
    if os.path.exists(DLIB_MODEL_PATH) and os.path.getsize(DLIB_MODEL_PATH) > 90_000_000:
        return
    import bz2
    import urllib.request

    compressed_path = DLIB_MODEL_PATH + ".bz2"
    urllib.request.urlretrieve(DLIB_MODEL_URL, compressed_path)
    with open(compressed_path, "rb") as f_in:
        data = bz2.decompress(f_in.read())
    with open(DLIB_MODEL_PATH, "wb") as f_out:
        f_out.write(data)
    os.unlink(compressed_path)


@st.cache_resource
def _get_dlib_models():
    _ensure_dlib_model()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    return detector, predictor


def detect_landmarks(img_float_bgr):
    """Restituisce (68,2) landmark in coordinate pixel, o None se nessun volto
    o se il modello dlib non e' disponibile (import fallito o download fallito)."""
    if not DLIB_OK:
        return None
    try:
        detector, predictor = _get_dlib_models()
    except Exception as exc:
        st.warning(
            f"Impossibile caricare il modello dei landmark del volto: {exc} / "
            f"Could not load the face landmark model: {exc}"
        )
        return None
    img_u8 = (np.clip(img_float_bgr, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    # primo tentativo veloce (upsample=1); se non trova nulla, riprova con
    # upsample piu' aggressivo (piu' lento ma piu' robusto su crop stretti
    # o volti piccoli nel frame, es. aspect ratio molto larghi/stretti)
    faces = detector(gray, 1)
    if len(faces) == 0:
        faces = detector(gray, 2)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return pts


def detect_landmarks_at_resolution(img_path, target_w, target_h):
    """Rileva i landmark sull'immagine ORIGINALE intera (il rilevamento e'
    molto piu' affidabile a piena inquadratura che su crop stretti/piccoli),
    poi proietta i punti nelle coordinate del crop+resize finale."""
    orig = cv2.imread(img_path)
    if orig is None:
        return None
    oh, ow = orig.shape[:2]
    orig_float = orig.astype(np.float32) / 255.0
    pts = detect_landmarks(orig_float)
    if pts is None:
        return None

    x0, y0, crop_w, crop_h = compute_fit_aspect_crop(ow, oh, target_w, target_h)
    scale = np.array([target_w / crop_w, target_h / crop_h], dtype=np.float32)
    pts_mapped = (pts - np.array([x0, y0], dtype=np.float32)) * scale
    return pts_mapped.astype(np.float32)



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
    """Envelope di energia bassi/medi/alti via STFT + frame dei beat + BPM."""
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


MAJOR_KEY_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MAJOR_KEY_PROFILE = MAJOR_KEY_PROFILE / MAJOR_KEY_PROFILE.sum()
MINOR_KEY_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MINOR_KEY_PROFILE = MINOR_KEY_PROFILE / MINOR_KEY_PROFILE.sum()


def analyze_audio_character(path, duration_sec):
    """Estrae due descrittori GLOBALI del brano (una volta sola, non per
    frame), usati per cambiare il "carattere" della deformazione da brano a
    brano:
    - mode_score: -1 (tonalita' minore) .. +1 (tonalita' maggiore), via
      correlazione del chroma medio contro i profili di Krumhansl-Kessler
    - complexity_score: 0..1, numero di picchi distinti nello spettro medio
      (proxy del numero di "voci"/strumenti simultanei nel mix - un accordo
      denso o un mix con molti strumenti produce piu' picchi spettrali di
      una linea di basso e una batteria sole)
    """
    y, sr = librosa.load(path, sr=22050, mono=True, duration=duration_sec)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = chroma.mean(axis=1)
    mean_chroma = mean_chroma / (mean_chroma.sum() + 1e-9)

    best_major, best_minor = -2.0, -2.0
    for shift in range(12):
        corr_maj = np.corrcoef(mean_chroma, np.roll(MAJOR_KEY_PROFILE, shift))[0, 1]
        corr_min = np.corrcoef(mean_chroma, np.roll(MINOR_KEY_PROFILE, shift))[0, 1]
        best_major = max(best_major, corr_maj)
        best_minor = max(best_minor, corr_min)
    mode_score = float(np.clip(best_major - best_minor, -1.0, 1.0))

    stft = np.abs(librosa.stft(y, n_fft=2048))
    mean_spec = stft.mean(axis=1)
    mean_spec_db = librosa.amplitude_to_db(mean_spec, ref=max(mean_spec.max(), 1e-9))
    peaks, _ = find_peaks(mean_spec_db, height=-30, distance=5)
    complexity_score = float(np.clip(len(peaks) / 30.0, 0.0, 1.0))

    return mode_score, complexity_score


def synthetic_bands(target_fps, duration_sec):
    """Fallback senza audio: tre curve ADSR leggermente sfasate tra loro."""
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

def build_face_mesh_points(pts, shape):
    """Landmark + punti sul contorno (hull espanso) per estendere la mesh
    oltre il volto stretto, cosi' il warp non si ferma bruscamente al bordo."""
    h, w = shape[:2]
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)
    hull_center = hull.mean(axis=0)
    hull_expanded = (hull_center + (hull - hull_center) * 1.35).astype(np.float32)
    hull_expanded[:, 0] = np.clip(hull_expanded[:, 0], 1, w - 2)
    hull_expanded[:, 1] = np.clip(hull_expanded[:, 1], 1, h - 2)

    all_pts = np.concatenate([pts, hull_expanded], axis=0).astype(np.float32)
    all_pts[:, 0] = np.clip(all_pts[:, 0], 1, w - 2)
    all_pts[:, 1] = np.clip(all_pts[:, 1], 1, h - 2)
    return all_pts, hull_expanded


def get_delaunay_triangle_indices(rect, points):
    """Triangolazione Delaunay -> lista di triple di INDICI (non coordinate)."""
    subdiv = cv2.Subdiv2D(rect)
    pts_list = [(float(p[0]), float(p[1])) for p in points]
    for p in pts_list:
        subdiv.insert(p)

    coord_to_idx = {(round(p[0], 1), round(p[1], 1)): i for i, p in enumerate(pts_list)}
    triangles_idx = []
    for t in subdiv.getTriangleList():
        # cast esplicito a float nativo Python: t arriva come numpy.float32,
        # e round() su un numpy scalar resta un numpy scalar - confrontarlo
        # con le chiavi (float nativo) del dizionario fallisce SEMPRE anche
        # quando i valori "sembrano" uguali (precisione binaria diversa tra
        # float32 e float64), azzerando silenziosamente tutti i triangoli.
        tri_pts = [(float(t[0]), float(t[1])), (float(t[2]), float(t[3])),
                   (float(t[4]), float(t[5]))]
        idx = [coord_to_idx[(round(tp[0], 1), round(tp[1], 1))] for tp in tri_pts
               if (round(tp[0], 1), round(tp[1], 1)) in coord_to_idx]
        if len(idx) == 3:
            triangles_idx.append(tuple(idx))
    return triangles_idx


def warp_triangle(src_img, dst_img, t_src, t_dst):
    """Trasforma affine un singolo triangolo da src a dst e lo fonde in dst_img."""
    h_img, w_img = dst_img.shape[:2]
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    x2, y2, w2, h2 = r2
    x2c, y2c = max(x2, 0), max(y2, 0)
    w2c = min(x2 + w2, w_img) - x2c
    h2c = min(y2 + h2, h_img) - y2c
    if r1[2] <= 0 or r1[3] <= 0 or w2c <= 0 or h2c <= 0:
        return

    t1_rect = [(t_src[i][0] - r1[0], t_src[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]) for i in range(3)]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), cv2.LINE_AA)

    img1_rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if img1_rect.size == 0:
        return
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, (r2[2], r2[3]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
    img2_rect = img2_rect * mask

    oy0, ox0 = y2c - r2[1], x2c - r2[0]
    img2_rect_c = img2_rect[oy0:oy0 + h2c, ox0:ox0 + w2c]
    mask_c = mask[oy0:oy0 + h2c, ox0:ox0 + w2c]

    dst_slice = dst_img[y2c:y2c + h2c, x2c:x2c + w2c]
    dst_slice[:] = dst_slice * (1 - mask_c) + img2_rect_c


def warp_face_mesh(base_img, all_src_pts, displaced_landmarks, hull_expanded, triangles_idx):
    all_dst_pts = np.concatenate([displaced_landmarks, hull_expanded], axis=0).astype(np.float32)
    output = base_img.copy()
    for idx in triangles_idx:
        t_src = [tuple(all_src_pts[i]) for i in idx]
        t_dst = [tuple(all_dst_pts[i]) for i in idx]
        warp_triangle(base_img, output, t_src, t_dst)
    return output


def compute_smile_score(pts):
    """Curvatura della bocca nella foto ORIGINALE (dai landmark, non da Haar
    Cascade - piu' preciso): positivo se sorride, vicino a zero/negativo se
    neutra o accigliata. Usata per decidere se esagerare il sorriso in un
    ghigno horror o forzare la bocca in un urlo."""
    left_corner, right_corner = pts[48], pts[54]
    top_lip, bottom_lip = pts[51], pts[57]
    mouth_width = float(np.linalg.norm(right_corner - left_corner)) + 1e-6
    corner_avg_y = (left_corner[1] + right_corner[1]) / 2.0
    center_y = (top_lip[1] + bottom_lip[1]) / 2.0
    lift = center_y - corner_avg_y
    return float(np.clip(lift / mouth_width * 3.0, -1.0, 1.0))


def build_dynamic_displacement(pts, jaw_i, mouth_i, eye_i, eye_jitter, rng, smile_bias=0.0):
    """Sposta i landmark per gruppo. eye_jitter aggiunge un tremore casuale
    ad alta frequenza (ricalcolato ogni frame) sugli occhi, per un carattere
    "nervoso" distinto dalla crescita liscia degli altri gruppi. smile_bias
    (da compute_smile_score) esagera il sorriso originale in un ghigno se
    positivo, o forza la bocca in un urlo verticale se neutro/negativo."""
    displaced = pts.copy()

    eye_r_center = pts[LANDMARK_GROUPS["eye_r"]].mean(axis=0)
    for i in LANDMARK_GROUPS["eye_r"]:
        base = pts[i] + (pts[i] - eye_r_center) * (0.9 * eye_i)
        displaced[i] = base + rng.uniform(-1, 1, 2) * eye_jitter * 7.0

    eye_l_center = pts[LANDMARK_GROUPS["eye_l"]].mean(axis=0)
    for i in LANDMARK_GROUPS["eye_l"]:
        base = pts[i] + (pts[i] - eye_l_center) * (0.9 * eye_i)
        displaced[i] = base + rng.uniform(-1, 1, 2) * eye_jitter * 7.0

    smile_pos = max(smile_bias, 0.0)
    smile_neg = max(-smile_bias, 0.0)
    mouth_center = pts[LANDMARK_GROUPS["mouth"]].mean(axis=0)
    for i in LANDMARK_GROUPS["mouth"]:
        dx = (pts[i][0] - mouth_center[0]) * (1.1 * mouth_i) * (1.0 + smile_pos * 0.9)
        dy = 16.0 * mouth_i * (1.0 - smile_pos * 0.3) + smile_neg * mouth_i * 12.0
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


def mid_gate(x, center=0.5, sharpness=12.0):
    """Soglia morbida (sigmoide): la bocca resta quasi chiusa finche' i medi
    non superano la soglia, poi scatta - non una sfumatura continua."""
    return float(1.0 / (1.0 + np.exp(-sharpness * (x - center))))


def apply_bulge_roi(img, center, radius, strength):
    """Lente di ingrandimento radiale: strength>0 campiona un'area piu'
    piccola vicino al centro (la feature sembra piu' grande, es. occhio che
    si dilata); strength<0 fa l'opposto (si rimpicciolisce/risucchia).
    Applicata solo nel riquadro locale attorno al centro per velocita'."""
    if radius <= 1 or abs(strength) < 1e-4:
        return img
    h, w = img.shape[:2]
    cx, cy = center
    x0, x1 = int(max(cx - radius, 0)), int(min(cx + radius, w))
    y0, y1 = int(max(cy - radius, 0)), int(min(cy + radius, h))
    if x1 <= x0 or y1 <= y0:
        return img

    sub = img[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    local_cx, local_cy = cx - x0, cy - y0

    xx, yy = np.meshgrid(np.arange(sw).astype(np.float32), np.arange(sh).astype(np.float32))
    dx = xx - local_cx
    dy = yy - local_cy
    d = np.sqrt(dx * dx + dy * dy)
    norm = np.clip(d / radius, 0, 1)

    factor = 1.0 - strength * (1.0 - norm) ** 2
    factor = np.where(d < radius, factor, 1.0)
    d_safe = np.where(d < 1e-6, 1.0, d)
    new_d = factor * d

    map_x = np.where(d < 1e-6, local_cx, local_cx + (dx / d_safe) * new_d)
    map_y = np.where(d < 1e-6, local_cy, local_cy + (dy / d_safe) * new_d)
    map_x = np.clip(map_x, 0, sw - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, sh - 1).astype(np.float32)

    warped_sub = cv2.remap(sub, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    out = img.copy()
    out[y0:y1, x0:x1] = warped_sub
    return out


def apply_directional_stretch(img, center, radius, stretch_x, stretch_y):
    """Allungamento/allargamento anisotropo (assi x/y indipendenti) attorno
    a un centro, con dissolvenza verso il bordo del raggio - stessa
    struttura di apply_bulge_roi ma senza vincolo di simmetria radiale,
    per un volto che si allunga in verticale o si allarga in orizzontale
    invece di gonfiarsi in modo uniforme in tutte le direzioni."""
    if radius <= 1 or (abs(stretch_x) < 1e-4 and abs(stretch_y) < 1e-4):
        return img
    h, w = img.shape[:2]
    cx, cy = center
    x0, x1 = int(max(cx - radius, 0)), int(min(cx + radius, w))
    y0, y1 = int(max(cy - radius, 0)), int(min(cy + radius, h))
    if x1 <= x0 or y1 <= y0:
        return img

    sub = img[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    local_cx, local_cy = cx - x0, cy - y0

    xx, yy = np.meshgrid(np.arange(sw).astype(np.float32), np.arange(sh).astype(np.float32))
    dx = xx - local_cx
    dy = yy - local_cy
    d = np.sqrt(dx * dx + dy * dy) + 1e-6
    norm = np.clip(d / radius, 0, 1)
    fall = (1.0 - norm) ** 2

    fx = 1.0 - float(np.clip(stretch_x, -0.9, 0.9)) * fall
    fy = 1.0 - float(np.clip(stretch_y, -0.9, 0.9)) * fall

    map_x = np.clip(local_cx + dx * fx, 0, sw - 1).astype(np.float32)
    map_y = np.clip(local_cy + dy * fy, 0, sh - 1).astype(np.float32)
    warped_sub = cv2.remap(sub, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    out = img.copy()
    out[y0:y1, x0:x1] = warped_sub
    return out


def render_anatomical_warp(base_img, pts, env_bass, env_mid, env_high, beat_frames,
                            seed, base_intensity, w_bass, w_mid, w_high, growth_rate,
                            writer, mode_score=0.0, complexity_score=0.5,
                            smile_override=None):
    """Deforma il volto su una mesh triangolata (Delaunay) ancorata ai landmark,
    piu' dilatazione radiale (bulge) su occhi e viso intero e uno stretch
    direzionale legato al carattere del brano:
    - mode_score (-1 minore .. +1 maggiore): minore = il viso si allunga in
      verticale (droop/melt), maggiore = si allarga in orizzontale (swell)
    - complexity_score (0..1, densita' spettrale/n. voci nel mix): scala il
      caos/asimmetria e il tremore, cosi' brani densi risultano visibilmente
      piu' frammentati di brani minimali
    - smile_override: se non None (da -1 a 1), sovrascrive il rilevamento
      automatico del sorriso dalla foto (ghigno se positivo, urlo se negativo)
    Le tre bande hanno un carattere qualitativamente diverso, non solo
    un'intensita' diversa: bassi = colpo secco sulla mascella, medi = bocca
    che scatta aperta/chiusa a soglia, alti = tremore rapido sugli occhi."""
    rng = np.random.default_rng(seed)
    all_src_pts, hull_expanded = build_face_mesh_points(pts, base_img.shape)
    h, w = base_img.shape[:2]
    triangles_idx = get_delaunay_triangle_indices((0, 0, w, h), all_src_pts)

    eye_r_center = tuple(pts[LANDMARK_GROUPS["eye_r"]].mean(axis=0))
    eye_l_center = tuple(pts[LANDMARK_GROUPS["eye_l"]].mean(axis=0))
    eye_width = float(np.linalg.norm(pts[36] - pts[39]))
    eye_radius = max(eye_width * 1.6, 12.0)

    face_center = tuple(pts.mean(axis=0))
    face_radius = max(float(np.ptp(pts[:, 0])) * 0.8, 30.0)
    stretch_radius = max(float(np.ptp(pts[:, 0])) * 1.7, 60.0)

    # asimmetria fissa per questo render (seedata), tanto piu' marcata quanto
    # piu' il brano e' denso/complesso - rende ogni lato leggermente diverso
    asym_l, asym_r = 1.0 + rng.uniform(-0.3, 0.3, 2) * complexity_score
    smile_bias = compute_smile_score(pts) if smile_override is None else float(smile_override)

    total_frames = len(env_bass)
    growth_acc = 0.0

    for f in range(total_frames):
        eb, em, eh_ = float(env_bass[f]), float(env_mid[f]), float(env_high[f])
        avg_e = (eb + em + eh_) / 3.0
        growth_acc = min(1.0, growth_acc + (avg_e / total_frames) * growth_rate)

        permanent = growth_acc * base_intensity * 0.6

        # bassi: colpo secco strutturale, molto piu' forte sul beat
        jaw_i = permanent + eb * w_bass * base_intensity
        if f in beat_frames:
            jaw_i *= 2.2

        # medi: la bocca scatta aperta a soglia, non scala in modo lineare
        mouth_i = permanent * 0.4 + w_mid * base_intensity * mid_gate(em)

        # alti: componente liscia ridotta + tremore ad alta frequenza vero,
        # amplificato dalla complessita' spettrale del brano
        eye_i = permanent * 0.3 + eh_ * w_high * base_intensity * 0.4
        eye_jitter = eh_ * w_high * (0.6 + complexity_score * 0.8)

        displaced = build_dynamic_displacement(pts, jaw_i, mouth_i, eye_i, eye_jitter, rng,
                                                smile_bias=smile_bias)
        # leggera asimmetria sinistra/destra sulla mascella, seedata per brano
        for i in LANDMARK_GROUPS["jaw"][:9]:
            displaced[i] = pts[i] + (displaced[i] - pts[i]) * asym_l
        for i in LANDMARK_GROUPS["jaw"][9:]:
            displaced[i] = pts[i] + (displaced[i] - pts[i]) * asym_r

        frame = warp_face_mesh(base_img, all_src_pts, displaced, hull_expanded, triangles_idx)

        # dilatazione radiale vera (lente d'ingrandimento), non solo
        # spostamento di landmark: occhi simmetrici pilotati dagli alti,
        # viso intero che si gonfia pilotato dalla crescita permanente + bassi
        eye_bulge = float(np.clip(0.25 + eye_i * 0.9, 0.0, 0.9))
        frame = apply_bulge_roi(frame, eye_r_center, eye_radius, eye_bulge)
        frame = apply_bulge_roi(frame, eye_l_center, eye_radius, eye_bulge)

        face_bulge = float(np.clip(permanent * 0.7 + eb * 0.3, -0.9, 0.9))
        frame = apply_bulge_roi(frame, face_center, face_radius, face_bulge)

        # stretch direzionale: la tonalita' del brano decide se il viso si
        # allunga (minore, malinconico/orrido) o si allarga (maggiore,
        # gonfio/aggressivo) - cresce nel tempo con la progressione permanente
        stretch_amount = permanent * 0.8
        if mode_score < 0:
            frame = apply_directional_stretch(frame, face_center, stretch_radius,
                                               0.0, stretch_amount * (1.0 + abs(mode_score)))
        else:
            frame = apply_directional_stretch(frame, face_center, stretch_radius,
                                               stretch_amount * (1.0 + mode_score), 0.0)

        frame = clinical_grade(frame)
        frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.write(frame_u8)
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


def compute_voronoi_cells(img, region_mask, n_points, rng):
    """Parte costosa (Canny + query cKDTree su ogni pixel): va ricalcolata
    solo quando cambiano i punti seme (cioe' ai beat), non ad ogni frame."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    points = seed_points_in_region(gray_blur, region_mask, n_points, rng)
    labels = build_cell_labels((h, w), points)
    return points, labels


def apply_voronoi_displacement(img, points, labels, intensity, rng, mode_score=0.0,
                                complexity_score=0.5):
    """Parte economica (solo spostamento delle celle gia' segmentate): questa
    si ricalcola ad ogni frame perche' l'intensita' cambia con l'audio.
    mode_score: minore = i pezzi cadono di piu' (droop), maggiore = si
    spingono di piu' verso l'esterno (radiale/esplosivo). complexity_score
    scala il rumore casuale per pezzo, per un caos maggiore su brani densi."""
    h, w = img.shape[:2]
    n_cells = len(points)
    cx = points[:, 0].mean()
    cy = points[:, 1].mean()

    fall_bias = 0.7 + max(-mode_score, 0.0) * 0.6
    radial_bias = 1.2 + max(mode_score, 0.0) * 0.7
    jitter_scale = 0.5 + complexity_score

    disp_x = np.zeros(n_cells, dtype=np.float32)
    disp_y = np.zeros(n_cells, dtype=np.float32)
    for i, (px, py) in enumerate(points):
        dx, dy = px - cx, py - cy
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        push = (dist / max(w, h)) * radial_bias
        disp_x[i] = (dx / dist) * push * intensity * 45
        disp_y[i] = (dy / dist) * push * intensity * 45 + fall_bias * intensity * 30

    disp_x += rng.uniform(-6, 6, n_cells) * intensity * jitter_scale
    disp_y += rng.uniform(-3, 10, n_cells) * intensity * jitter_scale

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
                    seed, base_intensity, n_points, growth_rate, writer,
                    mode_score=0.0, complexity_score=0.5):
    """Scrive ogni frame direttamente su `writer` (niente accumulo in RAM).
    I punti/celle Voronoi si ricalcolano solo ai beat (costoso), non ogni
    frame (economico): stesso identico risultato visivo, molto meno CPU."""
    total_frames = len(env_bass)
    growth_acc = 0.0
    seed_offset = 0

    cache_rng = np.random.default_rng(seed)
    points, labels = compute_voronoi_cells(base_img, region_mask, n_points, cache_rng)

    for f in range(total_frames):
        eb, em, eh_ = float(env_bass[f]), float(env_mid[f]), float(env_high[f])
        avg_e = (eb + em + eh_) / 3.0
        growth_acc = min(1.0, growth_acc + (avg_e / total_frames) * growth_rate)

        if f in beat_frames:
            seed_offset += 1
            cache_rng = np.random.default_rng(seed + seed_offset)
            points, labels = compute_voronoi_cells(base_img, region_mask, n_points, cache_rng)

        # bassi: intensita' della frattura, con colpo secco sul beat
        intensity = 0.1 + base_intensity * (growth_acc * 0.5 + eb * 0.5)
        if f in beat_frames:
            intensity *= 1.7

        disp_rng = np.random.default_rng(seed + seed_offset * 1000 + f)
        frame = apply_voronoi_displacement(base_img, points, labels, intensity, disp_rng,
                                            mode_score=mode_score,
                                            complexity_score=complexity_score)
        frame = clinical_grade(frame)
        frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.write(frame_u8)


# ---------------------------------------------------------------------------
# STILE: CAPILLARY BLEED (rifatto: ancorato ai landmark, reattivo a bande)
# ---------------------------------------------------------------------------

def render_capillary_bleed(base_img, region_mask, pts, env_bass, env_mid, env_high,
                            beat_frames, seed, base_intensity, writer, n_walkers=40,
                            mode_score=0.0, complexity_score=0.5):
    """Scrive ogni frame direttamente su `writer` (niente accumulo in RAM).
    mode_score: minore = le venature derivano verso il basso (sanguinamento),
    maggiore = si diramano piu' verso l'esterno. complexity_score scala la
    frequenza di ramificazione, per venature piu' fitte su brani densi."""
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

    # deriva costante legata alla tonalita': minore = venature che colano
    # verso il basso, maggiore = si diramano piu' verso l'esterno
    drift = np.array([0.0, max(-mode_score, 0.0) * 0.5], dtype=np.float32)
    outward_bias = max(mode_score, 0.0) * 0.4

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
        outward = (walker_pos - np.array([w / 2, h / 2])) * outward_bias / max(w, h)
        walker_dir = (0.7 * walker_dir + 0.3 * np.stack([tangent_x, tangent_y], axis=1)
                      + rand_perturb + drift + outward)
        norm = np.linalg.norm(walker_dir, axis=1, keepdims=True) + 1e-6
        walker_dir = walker_dir / norm

        walker_pos = walker_pos + walker_dir * step_size
        walker_pos[:, 0] = np.clip(walker_pos[:, 0], 0, w - 1)
        walker_pos[:, 1] = np.clip(walker_pos[:, 1], 0, h - 1)

        xi2 = walker_pos[:, 0].astype(int)
        yi2 = walker_pos[:, 1].astype(int)
        vein_mask[yi2, xi2] = 1.0

        # gli ALTI pilotano la frequenza di ramificazione, amplificata dalla
        # complessita' spettrale (brani densi = venature piu' fitte)
        branch_prob = eh_ * 0.5 * (0.6 + complexity_score)
        if (f in beat_frames or rng.uniform(0, 1) < branch_prob) and len(walker_pos) < max_walkers:
            n_new = min(int(4 * (0.6 + complexity_score)), n_walkers)
            branch_idx = rng.integers(0, len(walker_pos), n_new)
            new_pos = walker_pos[branch_idx] + rng.uniform(-2, 2, (n_new, 2))
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0, w - 1)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0, h - 1)
            new_dir = rng.uniform(-1, 1, (n_new, 2)).astype(np.float32)
            walker_pos = np.concatenate([walker_pos, new_pos.astype(np.float32)], axis=0)
            walker_dir = np.concatenate([walker_dir, new_dir], axis=0)

        vein_blur = cv2.GaussianBlur(vein_mask, (3, 3), 0)
        vein_blur = np.clip(vein_blur * region_mask, 0, 1)

        # i BASSI pilotano lo spessore/opacita', con un colpo secco sul beat
        eb_effective = eb * 1.6 if f in beat_frames else eb
        alpha = vein_blur[..., None] * (0.4 + 0.5 * eb_effective) * base_intensity
        frame = base_img * (1 - alpha) + bleed_color * alpha
        frame = clinical_grade(frame)
        frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.write(frame_u8)


# ---------------------------------------------------------------------------
# STILE: VORONOI + CAPILLARY (combo) - il volto si frattura in placche E
# sanguina lungo le crepe tra i pezzi (venature seminate sui bordi Voronoi
# invece che sui landmark del volto)
# ---------------------------------------------------------------------------

def crack_seeds_from_labels(labels, n_seeds, rng, w, h):
    gx = cv2.Sobel(labels.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(labels.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    boundary = (np.abs(gx) + np.abs(gy)) > 0
    ys, xs = np.where(boundary)
    if len(xs) == 0:
        return np.array([[w / 2, h / 2]], dtype=np.float32)
    idx = rng.choice(len(xs), min(n_seeds, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def render_voronoi_capillary_combo(base_img, region_mask, env_bass, env_mid, env_high,
                                    beat_frames, seed, base_intensity, n_points,
                                    growth_rate, writer, mode_score=0.0, complexity_score=0.5):
    h, w = base_img.shape[:2]
    total_frames = len(env_bass)
    growth_acc = 0.0
    seed_offset = 0
    rng = np.random.default_rng(seed)

    cache_rng = np.random.default_rng(seed)
    points, labels = compute_voronoi_cells(base_img, region_mask, n_points, cache_rng)

    n_walkers = max(n_points, 20)
    walker_pos = crack_seeds_from_labels(labels, n_walkers, rng, w, h)
    walker_dir = rng.uniform(-1, 1, walker_pos.shape).astype(np.float32)
    vein_mask = np.zeros((h, w), dtype=np.float32)
    bleed_color = np.array([0.05, 0.02, 0.35], dtype=np.float32)

    gray = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx_img = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=5)
    gy_img = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx_img ** 2 + gy_img ** 2) + 1e-6
    gx_n = gx_img / mag
    gy_n = gy_img / mag

    max_walkers = n_walkers * 4
    drift = np.array([0.0, max(-mode_score, 0.0) * 0.5], dtype=np.float32)
    outward_bias = max(mode_score, 0.0) * 0.4

    for f in range(total_frames):
        eb, em, eh_ = float(env_bass[f]), float(env_mid[f]), float(env_high[f])
        avg_e = (eb + em + eh_) / 3.0
        growth_acc = min(1.0, growth_acc + (avg_e / total_frames) * growth_rate)

        if f in beat_frames:
            seed_offset += 1
            cache_rng = np.random.default_rng(seed + seed_offset)
            points, labels = compute_voronoi_cells(base_img, region_mask, n_points, cache_rng)
            new_seeds = crack_seeds_from_labels(labels, 5, rng, w, h)
            walker_pos = np.concatenate([walker_pos, new_seeds], axis=0)
            walker_dir = np.concatenate(
                [walker_dir, rng.uniform(-1, 1, new_seeds.shape).astype(np.float32)], axis=0)

        frac_intensity = 0.08 + base_intensity * (growth_acc * 0.4 + eb * 0.4)
        if f in beat_frames:
            frac_intensity *= 1.6

        disp_rng = np.random.default_rng(seed + seed_offset * 1000 + f)
        fractured = apply_voronoi_displacement(base_img, points, labels, frac_intensity, disp_rng,
                                                mode_score=mode_score,
                                                complexity_score=complexity_score)

        step_size = 1.0 + 2.5 * em
        xi = np.clip(walker_pos[:, 0].astype(int), 0, w - 1)
        yi = np.clip(walker_pos[:, 1].astype(int), 0, h - 1)
        tangent_x = -gy_n[yi, xi]
        tangent_y = gx_n[yi, xi]
        rand_perturb = rng.uniform(-0.4, 0.4, walker_dir.shape).astype(np.float32)
        outward = (walker_pos - np.array([w / 2, h / 2])) * outward_bias / max(w, h)
        walker_dir = (0.7 * walker_dir + 0.3 * np.stack([tangent_x, tangent_y], axis=1)
                      + rand_perturb + drift + outward)
        norm = np.linalg.norm(walker_dir, axis=1, keepdims=True) + 1e-6
        walker_dir = walker_dir / norm
        walker_pos = walker_pos + walker_dir * step_size
        walker_pos[:, 0] = np.clip(walker_pos[:, 0], 0, w - 1)
        walker_pos[:, 1] = np.clip(walker_pos[:, 1], 0, h - 1)
        xi2 = walker_pos[:, 0].astype(int)
        yi2 = walker_pos[:, 1].astype(int)
        vein_mask[yi2, xi2] = 1.0

        branch_prob = eh_ * 0.4 * (0.6 + complexity_score)
        if (f in beat_frames or rng.uniform(0, 1) < branch_prob) and len(walker_pos) < max_walkers:
            n_new = min(int(3 * (0.6 + complexity_score)), n_walkers)
            branch_idx = rng.integers(0, len(walker_pos), n_new)
            new_pos = walker_pos[branch_idx] + rng.uniform(-2, 2, (n_new, 2))
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0, w - 1)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0, h - 1)
            new_dir = rng.uniform(-1, 1, (n_new, 2)).astype(np.float32)
            walker_pos = np.concatenate([walker_pos, new_pos.astype(np.float32)], axis=0)
            walker_dir = np.concatenate([walker_dir, new_dir], axis=0)

        vein_blur = cv2.GaussianBlur(vein_mask, (3, 3), 0)
        vein_blur = np.clip(vein_blur * region_mask, 0, 1)
        alpha = vein_blur[..., None] * (0.4 + 0.5 * eb) * base_intensity * 0.9
        frame = fractured * (1 - alpha) + bleed_color * alpha
        frame = clinical_grade(frame)
        frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.write(frame_u8)


# ---------------------------------------------------------------------------
# EXPORT VIDEO (streaming diretto su disco, mai l'intero video in RAM)
# ---------------------------------------------------------------------------

def open_video_writer(out_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (width, height))


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

STYLE_HASHTAGS = {
    STYLE_ANATOMICAL: "#anatomicalwarp",
    STYLE_VORONOI: "#voronoifracture",
    STYLE_CAPILLARY: "#capillarybleed",
    STYLE_COMBO: "#voronoicapillary",
}


def build_report(style_key, seed, duration_sec, fps, resolution, has_audio, bpm_value,
                  weights_used, vol_number):
    style_label = STYLE_LABELS[style_key]
    audio_line_it = f"BPM rilevato: {bpm_value:.1f}" if bpm_value else "Nessun audio: envelope sintetico"
    audio_line_en = f"Detected BPM: {bpm_value:.1f}" if bpm_value else "No audio: synthetic envelope"
    style_tag = STYLE_HASHTAGS.get(style_key, "")
    hashtags = (f"#generativeart #creativecoding #algorithmicart #puredsp #dlib "
                f"{style_tag} #hyperrealism #uncannyvalley #clinicalsurrealism "
                f"#audiovisualart #techno")

    report = f"""
[IT]
[BodyError] //  Vol. {vol_number:03d}
:: REPORT
Stile: {style_label}
Seed: {seed}
Durata: {duration_sec:.1f}s @ {fps}fps
Risoluzione: {resolution}
Audio: {"presente" if has_audio else "assente"} — {audio_line_it}
Pesi banda (bassi/medi/alti): {weights_used}
Tecnica: landmark del volto (dlib, 68 punti) + DSP puro
:: Riferimenti estetici:
Plastinazione Anatomica
Iperrealismo Scultoreo Clinico
:: Reinterpretati come Processo Algoritmico.

Direction & Algorithm: Loop507

{hashtags}

[EN]
[BodyError] //  Vol. {vol_number:03d}
:: REPORT
Style: {style_label}
Seed: {seed}
Duration: {duration_sec:.1f}s @ {fps}fps
Resolution: {resolution}
Audio: {"present" if has_audio else "none"} — {audio_line_en}
Band weights (bass/mid/high): {weights_used}
Technique: face landmarks (dlib, 68 points) + pure DSP
:: Aesthetic references:
Anatomical Plastination
Clinical Hyperrealist Sculpture
:: Reinterpreted as an Algorithmic Process.

Direction & Algorithm: Loop507

{hashtags}
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
        st.session_state.render_count = 0

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
        aw_auto_smile = st.checkbox(
            "Rileva ghigno/urlo automaticamente dalla foto / "
            "Auto-detect grin/scream from photo", value=True, key="aw_auto_smile",
            help="Se disattivato, usa lo slider sotto invece di leggere "
                 "l'espressione dalla foto caricata. / If off, uses the "
                 "slider below instead of reading the expression from the "
                 "uploaded photo.",
        )
        aw_smile_manual = st.slider(
            "Forza urlo (-1) / ghigno (+1) / Force scream (-1) / grin (+1)",
            -1.0, 1.0, 0.0, 0.1, key="aw_smile_manual", disabled=aw_auto_smile,
        )
    with st.expander("Voronoi Fracture",
                      expanded=(style_key in (STYLE_VORONOI, STYLE_COMBO))):
        if style_key == STYLE_COMBO:
            st.caption("Usati anche dal combo Voronoi + Capillary. / "
                       "Also used by the Voronoi + Capillary combo.")
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

    col_render1, col_render2 = st.columns(2)
    with col_render1:
        render_clicked = st.button("Genera / Render", type="primary", key="button_render")
    with col_render2:
        preview_clicked = st.button(
            "Anteprima 5s alta risoluzione / 5s high-res preview", key="button_preview",
            help="Render breve (max 5s) ma alla risoluzione piena scelta sopra, "
                 "per vedere la qualita' reale prima del render completo. Costa "
                 "piu' CPU di un'anteprima a bassa risoluzione. / Short render "
                 "(max 5s) at the full chosen resolution, to check real quality "
                 "before the full render. Uses more CPU than a low-res preview.",
        )

    def do_render(render_w, render_h, render_duration, output_key, report_key,
                   progress_label_prefix="", is_official=True):
        if image_file is None:
            st.error("Carica una foto prima di procedere. / Upload a photo first.")
            return
        if not DLIB_OK:
            st.error("dlib non disponibile: impossibile procedere. / "
                      "dlib not available: cannot proceed.")
            return

        st.session_state[output_key] = None
        st.session_state[report_key] = None

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.jpg")
            with open(img_path, "wb") as fh:
                fh.write(image_file.getvalue())

            audio_path = None
            if audio_file is not None:
                audio_path = os.path.join(tmpdir, "input_audio.mp3")
                with open(audio_path, "wb") as fh:
                    fh.write(audio_file.getvalue())

            progress = st.progress(0, text=progress_label_prefix + "Analisi audio / Audio analysis...")

            bpm_value = None
            mode_score, complexity_score = 0.0, 0.5
            if audio_path is not None and LIBROSA_OK:
                env_bass, env_mid, env_high, beat_frames, bpm_value = analyze_audio_bands(
                    audio_path, fps, render_duration)
                mode_score, complexity_score = analyze_audio_character(audio_path, render_duration)
            else:
                if audio_path is not None and not LIBROSA_OK:
                    st.warning("librosa non disponibile: uso envelope sintetico. / "
                               "librosa unavailable: using synthetic envelope.")
                env_bass, env_mid, env_high, beat_frames, bpm_value = synthetic_bands(
                    fps, render_duration)

            progress.progress(15, text=progress_label_prefix + "Caricamento immagine / Loading image...")
            base_img = load_image_fit_aspect(img_path, render_w, render_h)

            progress.progress(25, text=progress_label_prefix + "Rilevamento volto / Face detection...")
            pts = detect_landmarks_at_resolution(img_path, render_w, render_h)
            if pts is None:
                st.warning(
                    "Nessun volto rilevato: gli stili basati sui landmark saranno "
                    "limitati. / No face detected: landmark-based styles will be "
                    "limited."
                )

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

            progress.progress(35, text=progress_label_prefix + "Rendering frame / Rendering frames...")
            t0 = time.time()

            raw_video_path = os.path.join(tmpdir, "raw.mp4")
            writer = open_video_writer(raw_video_path, fps, render_w, render_h)

            try:
                if style_key == STYLE_ANATOMICAL:
                    render_anatomical_warp(
                        base_img, pts, env_bass, env_mid, env_high, beat_frames, int(seed),
                        base_intensity=float(aw_intensity), w_bass=float(w_bass),
                        w_mid=float(w_mid), w_high=float(w_high),
                        growth_rate=float(aw_growth), writer=writer,
                        mode_score=mode_score, complexity_score=complexity_score,
                        smile_override=None if aw_auto_smile else float(aw_smile_manual),
                    )
                elif style_key == STYLE_VORONOI:
                    render_voronoi(
                        base_img, region_mask, env_bass, env_mid, env_high, beat_frames,
                        int(seed), base_intensity=float(vf_intensity) * (0.5 + w_bass * 0.5),
                        n_points=int(vf_points), growth_rate=float(vf_growth), writer=writer,
                        mode_score=mode_score, complexity_score=complexity_score,
                    )
                elif style_key == STYLE_CAPILLARY:
                    render_capillary_bleed(
                        base_img, region_mask, pts, env_bass, env_mid, env_high, beat_frames,
                        int(seed), base_intensity=float(cb_intensity), writer=writer,
                        n_walkers=int(cb_walkers),
                        mode_score=mode_score, complexity_score=complexity_score,
                    )
                elif style_key == STYLE_COMBO:
                    render_voronoi_capillary_combo(
                        base_img, region_mask, env_bass, env_mid, env_high, beat_frames,
                        int(seed), base_intensity=float(vf_intensity) * (0.5 + w_bass * 0.5),
                        n_points=int(vf_points), growth_rate=float(vf_growth), writer=writer,
                        mode_score=mode_score, complexity_score=complexity_score,
                    )
                else:
                    st.error("Stile non riconosciuto. / Unrecognized style.")
                    return
            finally:
                writer.release()

            elapsed = time.time() - t0
            progress.progress(70, text=f"{progress_label_prefix}Frame renderizzati in "
                                        f"{elapsed:.1f}s / Encoding...")

            progress.progress(85, text=progress_label_prefix + "Transcodifica H.264 + mux audio / "
                                        "H.264 transcode + audio mux...")
            final_path = os.path.join(tmpdir, "bodyerror_output.mp4")
            try:
                finalize_video(raw_video_path, audio_path, render_duration, final_path)
            except subprocess.CalledProcessError as exc:
                st.error(f"Errore ffmpeg / ffmpeg error: {exc.stderr.decode(errors='ignore')[-500:]}")
                return

            output_bytes_path = os.path.join(tempfile.gettempdir(),
                                              f"bodyerror_{output_key}_{int(time.time())}.mp4")
            with open(final_path, "rb") as src, open(output_bytes_path, "wb") as dst:
                dst.write(src.read())

            if is_official:
                st.session_state.render_count += 1
            vol_number = st.session_state.render_count if is_official else 0

            st.session_state[output_key] = output_bytes_path
            st.session_state[report_key] = build_report(
                style_key, int(seed), render_duration, fps,
                f"{render_w}x{render_h}",
                audio_path is not None, bpm_value,
                f"{w_bass:.1f} / {w_mid:.1f} / {w_high:.1f}",
                vol_number,
            )

            progress.progress(100, text=progress_label_prefix + "Completato / Done")

    if render_clicked:
        do_render(target_w, target_h, duration_sec, "output_path", "report_text")

    if preview_clicked:
        # anteprima breve (5s) ma alla risoluzione PIENA scelta sopra, cosi'
        # si vede davvero come verra' il render finale, non una versione
        # rimpicciolita. Costa piu' CPU di prima, ma e' quello che hai chiesto.
        do_render(target_w, target_h, min(duration_sec, 5), "preview_output_path",
                  "preview_report_text", progress_label_prefix="[Anteprima] ",
                  is_official=False)

    if st.session_state.get("preview_output_path") and os.path.exists(
            st.session_state["preview_output_path"]):
        st.caption("Anteprima 5s alta risoluzione / 5s high-res preview")
        st.video(st.session_state["preview_output_path"])

    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        st.video(st.session_state.output_path)

        # Due download indipendenti: cliccare uno dei due NON fa sparire
        # l'altro (entrambi leggono solo da session_state, che viene svuotato
        # esclusivamente all'avvio di un nuovo render, mai da un click di
        # download) - puoi scaricare video e report in qualsiasi ordine.
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with open(st.session_state.output_path, "rb") as fh:
                st.download_button(
                    "Scarica video / Download video", data=fh.read(),
                    file_name=f"bodyerror_vol{st.session_state.render_count:03d}.mp4",
                    mime="video/mp4", key="button_download_video",
                )
        with col_dl2:
            st.download_button(
                "Scarica report / Download report",
                data=st.session_state.report_text,
                file_name=f"bodyerror_report_vol{st.session_state.render_count:03d}.txt",
                mime="text/plain", key="button_download_report",
            )

        st.text_area("Report", st.session_state.report_text, height=280, key="area_report")


if __name__ == "__main__":
    main()
