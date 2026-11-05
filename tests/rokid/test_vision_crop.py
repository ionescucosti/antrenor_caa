"""Screen detection and cropping before the vision model, the model fallback
for locating the screen, the sharpness gate, the no-invention prompt and the
'question only' flow when the options are not readable."""
from __future__ import annotations

import json

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from helpers import device_headers, example_dataset_bytes, make_settings, upload_dataset
from rokid_backend.llm.mock import MockLLM
from rokid_backend.main import create_app
from rokid_backend.rag.imageprep import crop_image, find_screen, image_size, sharpness
from rokid_backend.rag.prompts import LOCATE_PROMPT, build_transcribe_prompt
from rokid_backend.rag.transcribe import TranscriptionError, prepare_for_vision, transcribe_image

QUESTION_BLOCK = "Which Python type is immutable?\nA. List\nB. Dictionary\nC. Tuple\nD. Set"


def _photo(width=1200, height=1600, screen=(300, 600, 800, 920), lamp=True, textured=True, background=40) -> bytes:
    """Synthetic room photo: dark background, a bright screen rectangle with dark
    'text' noise inside, and a small bright lamp at the top edge."""
    rng = np.random.default_rng(3)
    img = np.full((height, width), background, dtype=np.uint8)
    if screen:
        x0, y0, x1, y1 = screen
        img[y0:y1, x0:x1] = 235
        if textured:
            noise = rng.random((y1 - y0, x1 - x0)) < 0.08
            img[y0:y1, x0:x1][noise] = 30
    if lamp:
        img[5:60, width - 90:width - 20] = 255
    pix = fitz.Pixmap(fitz.csGRAY, width, height, img.tobytes(), False)
    return pix.tobytes("jpeg", jpg_quality=92)


class SpyLLM(MockLLM):
    """Keeps the bytes and prompts of every transcribe call."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[bytes, str, str]] = []

    def transcribe(self, image: bytes, mime: str, prompt: str) -> str:
        self.sent.append((image, mime, prompt))
        return super().transcribe(image, mime, prompt)


def test_find_screen_ignores_the_lamp_and_returns_the_screen_box():
    box = find_screen(_photo())
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 300 and y0 <= 600 and x1 >= 800 and y1 >= 920            # covers the screen (with a margin)
    assert x1 - x0 < 700 and y1 - y0 < 500 and y0 > 100                    # not the lamp, not the whole photo
    assert find_screen(_photo(screen=None, lamp=False)) is None            # nothing bright: no box
    assert find_screen(_photo(screen=None, lamp=True)) is None             # a small blob at the top is not a screen
    full = find_screen(_photo(width=1600, height=1200, screen=(0, 0, 1600, 1200), lamp=False))   # a landscape close-up
    assert full is not None and full[2] - full[0] >= 1550


def test_crop_image_and_sharpness():
    data = _photo()
    crop = crop_image(data, (300, 600, 800, 920))
    assert crop.startswith(b"\xff\xd8") and image_size(crop) == (500, 320)
    assert sharpness(crop) > 200                                            # text-like detail inside the screen
    assert sharpness(crop_image(_photo(textured=False), (300, 600, 800, 920))) < 5   # a flat screen has none
    assert sharpness(_photo(screen=None, lamp=False)) < 1


def test_prepare_for_vision_crops_the_screen_at_full_resolution(tmp_path):
    settings = make_settings(tmp_path, vision_max_side=1792)
    data = _photo(width=3000, height=4000, screen=(800, 1500, 2000, 2250))
    prepared, mime, info = prepare_for_vision(data, "image/jpeg", MockLLM(), settings)
    assert info["crop"] and mime == "image/jpeg"
    w, h = image_size(prepared)
    assert 1200 <= w <= 1400 and 750 <= h <= 900                             # the screen plus margins, not downscaled
    assert not info.get("resized")
    # cropping off: the whole photo, downscaled as before
    prepared, mime, info = prepare_for_vision(data, "image/jpeg", MockLLM(), make_settings(tmp_path, vision_crop="off"))
    assert "crop" not in info and info["resized"] and image_size(prepared)[1] == 1792
    # a screen filling the photo is not cropped
    prepared, mime, info = prepare_for_vision(_photo(screen=(0, 0, 1200, 1600), lamp=False), "image/jpeg", MockLLM(), settings)
    assert "crop" not in info


def test_model_locates_the_screen_when_the_detector_fails(tmp_path):
    settings = make_settings(tmp_path, vision_crop_model_fallback=True)
    dark = _photo(width=1600, height=2000, screen=(400, 800, 1000, 1200), background=60, lamp=False)
    # make the screen too dim for the detector but still a distinct region
    img = np.frombuffer(fitz.Pixmap(dark).samples, dtype=np.uint8).reshape(2000, 1600).copy()
    img[800:1200, 400:1000] = 140
    dim = fitz.Pixmap(fitz.csGRAY, 1600, 2000, img.tobytes(), False).tobytes("jpeg", jpg_quality=92)
    assert find_screen(dim) is None
    llm = SpyLLM()
    llm.transcriptions += [json.dumps({"x0": 0.25, "y0": 0.4, "x1": 0.625, "y1": 0.6}), QUESTION_BLOCK]
    text, method = transcribe_image(dim, llm, settings)
    assert text == QUESTION_BLOCK and method == "llm:mock"
    (thumb, _, prompt1), (crop, _, prompt2) = llm.sent
    assert prompt1 == LOCATE_PROMPT and max(image_size(thumb)) <= 768
    w, h = image_size(crop)
    assert 600 <= w <= 700 and 400 <= h <= 480 and "Never complete, guess" in prompt2
    # an unusable reply falls back to the whole photo
    llm = SpyLLM()
    llm.transcriptions += ["I cannot see a screen", QUESTION_BLOCK]
    assert transcribe_image(dim, llm, settings)[0] == QUESTION_BLOCK
    assert image_size(llm.sent[1][0]) == (1433, 1792)   # the whole photo, downscaled as usual


def test_blurry_photo_is_refused_before_the_model(tmp_path):
    settings = make_settings(tmp_path, vision_min_sharpness=10)
    llm = SpyLLM()
    llm.transcriptions.append(QUESTION_BLOCK)
    with pytest.raises(TranscriptionError, match="neclara"):
        transcribe_image(_photo(textured=False), llm, settings)
    assert llm.sent == []
    assert transcribe_image(_photo(), llm, settings)[0] == QUESTION_BLOCK


def test_prompt_forbids_inventing_and_allows_question_only():
    prompt = build_transcribe_prompt()
    assert "output only the question line" in prompt and "Never complete, guess or reconstruct" in prompt
    assert "an invented option is worse than no option" in prompt


def test_question_only_transcription_returns_the_answer_text(tmp_path):
    llm = SpyLLM()
    app = create_app(make_settings(tmp_path), llm=llm)
    with TestClient(app) as client:
        upload_dataset(client, "python.json", example_dataset_bytes())
        llm.transcriptions.append("Which Python type is immutable?")      # options were not readable
        r = client.post("/api/v1/questions/answer", files={"image": ("q.jpg", _photo(), "image/jpeg")},
                        data={"request_id": "img-1"}, headers=device_headers())
        assert r.json() == {"text": "Tuple", "note": "Optiunile nu au fost lizibile in poza; cauta pe ecran textul raspunsului afisat."}
        llm.transcriptions.append("Which Python type is immutable?")
        full = client.post("/api/v1/questions/answer?full=1", files={"image": ("q.jpg", _photo(), "image/jpeg")},
                           data={"request_id": "img-2"}, headers=device_headers()).json()
        assert full["status"] == "answered" and full["match_type"] == "exact_qa" and full["answer"]["option_labels"] == []
        assert full["answer"]["text"] == "Tuple" and "lizibile" in full["message"] and full["transcribed_text"] == "Which Python type is immutable?"
        # with readable options the letter comes back and there is no note
        llm.transcriptions.append(QUESTION_BLOCK)
        r = client.post("/api/v1/questions/answer", files={"image": ("q.jpg", _photo(), "image/jpeg")},
                        data={"request_id": "img-3"}, headers=device_headers())
        assert r.json() == {"label": "C", "text": "Tuple"}
