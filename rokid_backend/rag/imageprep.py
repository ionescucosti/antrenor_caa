"""Shrink a photo before it goes to the vision model: fewer bytes to upload,
fewer image tokens, and no second resize on the provider side. Uses PyMuPDF
(already a dependency), never Pillow. Any failure returns the original bytes."""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

RE_ENCODE_ABOVE = 1_500_000   # bytes: re-encode as JPEG even when no resize is needed
JPEG_QUALITY = 88             # text stays crisp; a 12 MP photo lands around 300-500 KB after downscaling


def exif_orientation(data: bytes) -> int:
    """EXIF orientation of a JPEG (1 = normal, also when absent or unreadable).
    Photos stored rotated with an orientation tag are left untouched upstream,
    because re-encoding would drop the tag and the model would see them sideways."""
    try:
        if not data.startswith(b"\xff\xd8"):
            return 1
        i = 2
        while i + 4 <= len(data) and data[i] == 0xFF:
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            length = int.from_bytes(data[i + 2:i + 4], "big")
            if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
                tiff = i + 10
                order = {b"II": "little", b"MM": "big"}.get(data[tiff:tiff + 2])
                if order is None:
                    return 1
                ifd = tiff + int.from_bytes(data[tiff + 4:tiff + 8], order)
                count = int.from_bytes(data[ifd:ifd + 2], order)
                for n in range(min(count, 64)):
                    entry = ifd + 2 + n * 12
                    if int.from_bytes(data[entry:entry + 2], order) == 0x0112:
                        return int.from_bytes(data[entry + 8:entry + 10], order) or 1
                return 1
            if marker == 0xDA:  # start of scan: no more headers
                break
            i += 2 + length
    except Exception:  # noqa: BLE001 - malformed header
        return 1
    return 1


def prepare_image(data: bytes, mime: str, max_side: int, quality: int = JPEG_QUALITY) -> tuple[bytes, str, dict]:
    """Returns (bytes, mime, info). Downscales when the longer side exceeds
    `max_side` pixels (0 = never) and re-encodes as JPEG when the file is large.
    Small images are returned unchanged, byte for byte."""
    info: dict = {"original_bytes": len(data), "resized": False}
    if max_side <= 0 and len(data) <= RE_ENCODE_ABOVE:
        return data, mime, info
    if mime == "image/jpeg" and exif_orientation(data) != 1:
        info["skipped"] = "exif-orientation"
        return data, mime, info
    started = time.perf_counter()
    try:
        import fitz  # PyMuPDF
        pix = fitz.Pixmap(data)
        info.update(width=pix.width, height=pix.height)
        longest = max(pix.width, pix.height)
        needs_resize = 0 < max_side < longest
        if not needs_resize and len(data) <= RE_ENCODE_ABOVE:
            return data, mime, info
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if pix.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if needs_resize:
            scale = max_side / longest
            pix = fitz.Pixmap(pix, max(1, int(pix.width * scale)), max(1, int(pix.height * scale)), None)
        out = pix.tobytes("jpeg", jpg_quality=quality)
        info.update(resized=needs_resize, width=pix.width, height=pix.height, bytes=len(out),
                    ms=round((time.perf_counter() - started) * 1000, 1))
        return out, "image/jpeg", info
    except Exception as exc:  # noqa: BLE001 - unsupported format, corrupt file...
        log.warning("image preparation skipped: %s", type(exc).__name__)
        info["skipped"] = type(exc).__name__
        return data, mime, info


# --- screen detection and cropping ------------------------------------------------
# Glasses photos show the whole room; the screen with the question is a small
# bright rectangle in them. Sending the full photo downscaled makes the text
# 4-7 px high and unreadable, so the bright screen is found and cropped at full
# resolution before the model sees it.

SCREEN_THRESHOLDS = (215, 195, 175, 160)   # brightness levels tried from the brightest down (screens are near white)
SCREEN_MIN_AREA = 0.015                     # fraction of the photo: the first pass ignores small bright blobs (lamps, reflections)
SCREEN_MIN_AREA_SMALL = 0.006               # second pass, when nothing bigger was found
SCREEN_SKIP_ABOVE = 0.90                    # a box covering this much of the photo is not worth cropping
SHARPNESS_WIDTH = 1200                      # sharpness is measured on a copy scaled to this long side, so values are comparable


def _gray(data: bytes):
    import fitz
    import numpy as np
    pix = fitz.Pixmap(data)
    if pix.n - pix.alpha >= 3:
        pix = fitz.Pixmap(fitz.csGRAY, pix)
    elif pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)


def _component_at(mask, seen, y: int, x: int):
    """Flood fill from (y, x): (pixel count, x0, y0, x1, y1)."""
    from collections import deque
    h, w = mask.shape
    queue = deque([(y, x)])
    seen[y, x] = True
    n, y0, y1, x0, x1 = 0, y, y, x, x
    while queue:
        cy, cx = queue.popleft()
        n += 1
        y0, y1, x0, x1 = min(y0, cy), max(y1, cy), min(x0, cx), max(x1, cx)
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return n, x0, y0, x1, y1


def _screen_like(component, mask_shape, min_area: float) -> bool:
    """Big enough, roughly landscape, well filled, not glued to the top edge, wide enough."""
    h, w = mask_shape
    n, x0, y0, x1, y1 = component
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    not_a_lamp = y0 > 0.03 * h or n >= 0.25 * h * w   # small blobs glued to the top edge are lamps; a big one is the screen itself
    return n >= min_area * h * w and 0.9 <= bw / bh <= 2.6 and n / (bw * bh) >= 0.5 and not_a_lamp and bw >= 0.08 * w


def _largest_component(mask, min_area: float):
    """Bounding box of the largest screen-like bright component of a boolean mask."""
    import numpy as np
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        component = _component_at(mask, seen, y, x)
        if _screen_like(component, mask.shape, min_area) and (best is None or component[0] > best[0]):
            best = component
    return best


def _grow(small, component, thresh: int, min_area: float):
    """A white question card found at a high threshold may sit on a light-grey
    page: lower the threshold step by step and take the component around the
    same centre while it still looks like a screen and did not merge with the
    wall (area growing more than 2.5x)."""
    import numpy as np
    for lower in SCREEN_THRESHOLDS:
        if lower >= thresh:
            continue
        mask = small > lower
        cy, cx = (component[2] + component[4]) // 2, (component[1] + component[3]) // 2
        if not mask[cy, cx]:
            break
        bigger = _component_at(mask, np.zeros_like(mask, dtype=bool), cy, cx)
        if bigger[0] > 2.5 * component[0] or not _screen_like(bigger, small.shape, min_area):
            break
        component, thresh = bigger, lower
    return component


def image_size(data: bytes) -> tuple[int, int]:
    import fitz
    pix = fitz.Pixmap(data)
    return pix.width, pix.height


def find_screen(data: bytes) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) in pixels around the bright screen or page, with a small
    margin, or None when no screen-like region stands out."""
    try:
        gray = _gray(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("screen detection skipped: %s", type(exc).__name__)
        return None
    h, w = gray.shape
    step = max(1, max(h, w) // 400)
    small = gray[::step, ::step]
    sh, sw = small.shape
    for min_area in (SCREEN_MIN_AREA, SCREEN_MIN_AREA_SMALL):
        for thresh in SCREEN_THRESHOLDS:
            component = _largest_component(small > thresh, min_area)
            if component is None:
                continue
            _, x0, y0, x1, y1 = _grow(small, component, thresh, min_area)
            mx, my = int((x1 - x0 + 1) * step * 0.04), int((y1 - y0 + 1) * step * 0.06)
            return (max(0, x0 * step - mx), max(0, y0 * step - my), min(w, (x1 + 1) * step + mx), min(h, (y1 + 1) * step + my))
    return None


def crop_image(data: bytes, box: tuple[int, int, int, int], quality: int = 90) -> bytes:
    """JPEG of the box region at full resolution."""
    import fitz
    pix = fitz.Pixmap(data)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if pix.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    rect = fitz.IRect(*box)
    clip = fitz.Pixmap(pix.colorspace, rect)
    clip.set_origin(rect.x0, rect.y0)
    clip.copy(pix, rect)
    return clip.tobytes("jpeg", jpg_quality=quality)


def sharpness(data: bytes) -> float:
    """Variance of the Laplacian on a copy scaled to SHARPNESS_WIDTH on the long
    side: motion blur and defocus push it towards zero (flat images too)."""
    import numpy as np
    gray = _gray(data).astype(np.float32)
    step = max(1, max(gray.shape) // SHARPNESS_WIDTH)
    a = gray[::step, ::step]
    h, w = a.shape
    if h >= 30 and w >= 30:
        a = a[h // 10:-(h // 10), w // 10:-(w // 10)]   # the interior: text, not the bezel edges of the crop
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return float(lap.var())
