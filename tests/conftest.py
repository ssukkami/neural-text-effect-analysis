import importlib.util
import math
import re
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app.py"


class MockPredictor:
    def __init__(self):
        self.conf_cols = [
            "followers",
            "following",
            "num_posts",
            "cat_tech",
        ]

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def predict(
        self,
        text,
        followers=None,
        following=None,
        num_posts=None,
        account_type=None,
        audience="generic",
        image_grade=None,
        description_grade=None,
        timestamp=None,
        extra_features=None,
    ):
        clean = self.clean_text(text)

        if not clean:
            return {
                "ok": False,
                "error": "EMPTY_TEXT_AFTER_CLEANING",
            }

        words = re.findall(r"[A-Za-z']+", clean)
        word_count = len(words)
        hashtag_count = clean.count("#")
        mention_count = clean.count("@")
        question_count = clean.count("?")
        exclamation_count = clean.count("!")

        has_stage_a = (
            followers is not None
            and following is not None
            and num_posts is not None
        )

        text_residual = 0.0
        text_residual += 0.005 * min(question_count, 3)
        text_residual += 0.003 * min(hashtag_count, 5)
        text_residual += 0.002 * min(exclamation_count, 5)
        text_residual += 0.001 * min(word_count / 10.0, 5)

        if "drop" in clean or "share" in clean or "comment" in clean:
            text_residual += 0.02

        if has_stage_a:
            baseline = 4.0
            predicted = baseline + text_residual
            prediction_mode = "full_prediction"
            engagement_score = max(0.0, min(100.0, predicted * 17.0))
        else:
            baseline = None
            predicted = None
            prediction_mode = "text_only_analysis"
            engagement_score = None

        text_effect_score = max(0.0, min(100.0, 50.0 + text_residual * 500.0))

        polarity = 0.2 if any(w in clean for w in ["best", "works", "live"]) else 0.0

        return {
            "ok": True,
            "prediction_mode": prediction_mode,
            "stage_a_status": {
                "usable": True,
                "quality": "complete" if has_stage_a else "partial",
                "missing_features": [],
                "filled_with_zero": [],
                "imputed_with_train_mean": [],
                "unsupported_features": [],
            },
            "model_output": {
                "predicted_engagement_log": (
                    round(float(predicted), 4) if predicted is not None else None
                ),
                "baseline_component": (
                    round(float(baseline), 4) if baseline is not None else None
                ),
                "text_residual_component": round(float(text_residual), 4),
            },
            "relative_scores": {
                "engagement_relative_score": (
                    round(float(engagement_score), 1)
                    if engagement_score is not None else None
                ),
                "text_effect_relative_score": round(float(text_effect_score), 1),
            },
            "attention_analysis": {
                "terms": [
                    {"term": "machine", "score": 100},
                    {"term": "learning", "score": 80},
                ],
                "note": "Mock attention output for functional tests.",
            },
            "auxiliary_analysis": {
                "language_quality_score": 75.0,
                "audience_fit_score": 50.0,
                "engagement_potential_score": 90.0,
                "recommendations": {
                    "reasons": ["Call-to-action markers are present."],
                    "improvements": ["Increase readability by shortening sentences."],
                },
                "profile": {
                    "readability": {
                        "flesch": 60.0,
                        "fk_grade": 8.0,
                        "smog": 7.0,
                        "ari": 8.0,
                    },
                    "style": {
                        "word_count": word_count,
                        "sentence_count": max(1, clean.count(".") + clean.count("?")),
                        "avg_sentence_len": float(word_count),
                        "lexical_diversity_ttr": 0.8,
                        "hashtag_count": hashtag_count,
                        "mention_count": mention_count,
                        "question_count": question_count,
                        "exclamation_count": exclamation_count,
                        "emoji_count": 0,
                        "cta_hits": 1 if "drop" in clean or "share" in clean else 0,
                        "direct_address_hits": 1 if "you" in clean else 0,
                    },
                    "sentiment": {
                        "polarity": polarity,
                        "pos_pct": max(0.0, polarity * 100.0),
                        "neg_pct": 0.0,
                        "neu_pct": 100.0 - max(0.0, polarity * 100.0),
                    },
                    "signals": {
                        "cta": "drop" in clean or "share" in clean,
                        "direct_address": "you" in clean,
                        "question": question_count > 0,
                        "hashtags_ok": 2 <= hashtag_count <= 8,
                    },
                },
            },
            "debug": {
                "used_clean_text": clean,
                "conf_cols": list(self.conf_cols),
            },
        }


def install_mock_modules():
    mock_inference = types.ModuleType("inference")
    mock_inference.Predictor = MockPredictor
    sys.modules["inference"] = mock_inference

    if "webview" not in sys.modules:
        mock_webview = types.ModuleType("webview")
        mock_webview.create_window = lambda *args, **kwargs: None
        mock_webview.start = lambda *args, **kwargs: None
        sys.modules["webview"] = mock_webview


def load_local_app():
    if not APP_FILE.exists():
        raise FileNotFoundError(f"Не знайдено app.py: {APP_FILE}")

    install_mock_modules()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    spec = importlib.util.spec_from_file_location("local_diploma_app", APP_FILE)
    module = importlib.util.module_from_spec(spec)

    sys.modules["local_diploma_app"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "app"):
        raise AttributeError("У app.py не знайдено Flask-змінну app.")

    return module.app


@pytest.fixture(scope="session")
def flask_app():
    app = load_local_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(flask_app):
    with flask_app.test_client() as client:
        yield client


@pytest.fixture(scope="session")
def endpoints():
    return {
        "single_text": "/api/analyze",
        "ab_compare": "/api/ab_compare",
        "dataset": "/api/upload_csv",
    }


def assert_finite_number(value, field_name: str):
    assert isinstance(value, (int, float)), f"{field_name} must be numeric"
    assert math.isfinite(value), f"{field_name} must be finite"


def require_key(data: dict, key: str):
    assert key in data, f"Missing field: {key}"
    return data[key]


def assert_status(response, expected_status: int):
    data = response.get_json(silent=True)
    assert response.status_code == expected_status, data
    return data