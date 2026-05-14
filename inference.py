import os
import re
import math
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob
import textstat
import emoji as emoji_lib  # FIX: використовується для emoji_count (відповідає тренуванню)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "model_assets")

CONF_A_SAVE    = os.path.join(ASSETS_DIR, "confounder_mlp_A_FINAL.pth")
SCALER_A_SAVE  = os.path.join(ASSETS_DIR, "confounder_scaler_A_FINAL.pkl")
ROBERTA_SAVE   = os.path.join(ASSETS_DIR, "text_effect_roberta_B_FINAL.pth")
FEATURE_SCHEMA = os.path.join(ASSETS_DIR, "feature_schema.json")


# =========================
# Models
# =========================
class ConfounderMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32),     nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TextEffectRoBERTa(nn.Module):
    def __init__(self, model_name: str = "roberta-base", return_attn: bool = True):
        super().__init__()
        self.return_attn = return_attn
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.text_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    @staticmethod
    def mean_pool(last_hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.return_attn
        )
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        t = self.text_head(pooled)
        pred = self.regressor(t).squeeze(-1)

        if self.return_attn:
            return pred, out.attentions[-1]
        return pred


# =========================
# Predictor
# =========================
class Predictor:
    def __init__(self, device: Optional[str] = None):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        for path in (CONF_A_SAVE, SCALER_A_SAVE, ROBERTA_SAVE):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing asset: {path}")

        scaler_data = joblib.load(SCALER_A_SAVE)
        self.scaler     = scaler_data["scaler"]
        self.conf_cols: List[str] = scaler_data["conf_cols"]

        self.tokenizer_name = "roberta-base"
        self.max_len        = 160  # відповідає MAX_LEN у тренуванні
        self.tokenizer      = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.model_A = ConfounderMLP(in_dim=len(self.conf_cols))
        self.model_A.load_state_dict(
            torch.load(CONF_A_SAVE, map_location="cpu"), strict=True
        )
        self.model_A.to(self.device).eval()

        self.model_B = TextEffectRoBERTa(self.tokenizer_name, return_attn=True)
        self.model_B.load_state_dict(
            torch.load(ROBERTA_SAVE, map_location="cpu"), strict=True
        )
        self.model_B.to(self.device).eval()

        self.schema          = self._load_schema(FEATURE_SCHEMA)
        self.index_cal       = self._load_index_calibration()
        self.text_effect_cal = self._load_text_effect_calibration()

        self._stop = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "your", "you", "are", "was", "were", "will", "have", "has", "had",
            "not", "but", "can", "our", "their", "they", "them", "its", "it's",
            "a", "an", "to", "of", "in", "on"
        }

        self._surface_features = {
            "text_len", "word_count",
            "hashtag_count", "mention_count",
            "exclamation_count", "question_count", "emoji_count",
        }

        self._time_features = {
            "month", "is_weekend", "dow_sin", "dow_cos",
        }

    # -------------------------
    # Utils
    # -------------------------
    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _load_schema(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _read_calibration_pair(
        self,
        section: Dict[str, Any],
        default_p05: float,
        default_p95: float,
    ) -> Dict[str, float]:
        """Safely read p05/p95 from schema.

        The schema may contain keys with null values. In that case direct
        float(None) raises TypeError, so values must be validated before use.
        """
        p05 = self._safe_float(section.get("p05"), default_p05)
        p95 = self._safe_float(section.get("p95"), default_p95)

        if abs(p95 - p05) <= 1e-9:
            return {"p05": float(default_p05), "p95": float(default_p95)}

        return {"p05": float(p05), "p95": float(p95)}

    def _load_index_calibration(self) -> Dict[str, float]:
        sch = self.schema or {}
        default = {"p05": 0.586, "p95": 3.581}

        for key in ("index_calibration", "calibration", "engagement_index_calibration"):
            section = sch.get(key)
            if isinstance(section, dict):
                cal = self._read_calibration_pair(section, default["p05"], default["p95"])
                if cal != default or (section.get("p05") is not None and section.get("p95") is not None):
                    return cal

        if isinstance(sch, dict) and ("p05" in sch or "p95" in sch):
            return self._read_calibration_pair(sch, default["p05"], default["p95"])

        if isinstance(sch.get("target_stats"), dict):
            mu = self._safe_float(sch["target_stats"].get("mean"), 0.0)
            sd = self._safe_float(sch["target_stats"].get("std"),  0.0)
            if sd > 1e-9:
                return {"p05": mu - 2.0 * sd, "p95": mu + 2.0 * sd}

        return default

    def _load_text_effect_calibration(self) -> Dict[str, float]:
        sch = self.schema or {}
        default = {"p05": -0.5, "p95": 0.5}

        for key in ("text_effect_calibration", "residual_calibration", "text_score_calibration"):
            section = sch.get(key)
            if isinstance(section, dict):
                cal = self._read_calibration_pair(section, default["p05"], default["p95"])
                if cal != default or (section.get("p05") is not None and section.get("p95") is not None):
                    return cal

        return default

    def _train_feature_mean(self, col: str, default: float = 0.0) -> float:
        """Return the training-set mean stored in StandardScaler for a feature.

        Missing non-text confounders must be imputed by the training mean, not by
        zero. In scaled space this gives z=0 and prevents artificial
        out-of-distribution vectors during manual inference.
        """
        try:
            idx = self.conf_cols.index(col)
            return float(self.scaler.mean_[idx])
        except Exception:
            return float(default)

    @staticmethod
    def _is_missing_value(x: Any) -> bool:
        if x is None:
            return True
        try:
            if isinstance(x, str):
                return x.strip() == ""
            v = float(x)
            return math.isnan(v) or math.isinf(v)
        except Exception:
            return False

    def _value_or_train_mean(self, col: str, original_value: Any, computed_value: float) -> float:
        if self._is_missing_value(original_value):
            return self._train_feature_mean(col, computed_value)
        return float(computed_value)

    def _engagement_log_bounds(self) -> Dict[str, float]:
        """Robust bounds for engagement-scale outputs.

        The model target is non-negative in the training data. The upper bound is
        derived from p05/p95 calibration with a controlled margin, so extreme MLP
        extrapolations do not leak into the UI as impossible values.
        """
        p05 = float(self.index_cal.get("p05", 0.586))
        p95 = float(self.index_cal.get("p95", 3.581))
        span = max(1e-6, p95 - p05)
        lower = 0.0
        upper = p95 + 1.5 * span
        return {"lower": float(lower), "upper": float(upper)}

    def _clamp_engagement_log(self, value: float) -> float:
        bounds = self._engagement_log_bounds()
        return float(np.clip(float(value), bounds["lower"], bounds["upper"]))

    def _scale_and_guard_stage_a_features(self, x_arr: np.ndarray) -> Dict[str, Any]:
        scaled_raw = self.scaler.transform(x_arr)
        z_limit = 5.0
        abs_z = np.abs(scaled_raw[0])
        ood_features = [
            {
                "feature": str(col),
                "z": round(float(z), 4),
                "raw_value": round(float(x_arr[0, i]), 6),
            }
            for i, (col, z) in enumerate(zip(self.conf_cols, scaled_raw[0]))
            if abs(float(z)) > z_limit
        ]
        scaled_clipped = np.clip(scaled_raw, -z_limit, z_limit)
        return {
            "scaled": scaled_clipped,
            "max_abs_z_raw": round(float(abs_z.max()), 4) if len(abs_z) else 0.0,
            "clipped": bool(len(ood_features) > 0),
            "ood_features": ood_features,
            "z_clip_limit": z_limit,
        }

    # -------------------------
    # Text preprocessing
    # FIX: додано .lower() — відповідає тренуванню (.str.lower().str.strip())
    # -------------------------
    def clean_text(self, text: str) -> str:
        t = str(text or "")
        t = t.replace("\u00a0", " ").replace("\u200b", " ").strip()

        if t.lower() in {"", "none", "null", "nan", "no description"}:
            return ""

        t = re.sub(r"(https?://\S+|www\.\S+)", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        # FIX: lowercase — тренування використовувало .str.lower()
        t = t.lower()
        return t

    # -------------------------
    # Timestamp parsing
    # -------------------------
    def _parse_timestamp(self, timestamp: Optional[str]) -> Optional[datetime]:
        if timestamp is None:
            return None

        s = str(timestamp).strip()
        if not s:
            return None

        s = s.replace("Z", "+00:00")

        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass

        return None

    # -------------------------
    # Stage A diagnostics
    # -------------------------
    def _collect_stage_a_diagnostics(
        self,
        followers:         Optional[float],
        following:         Optional[float],
        num_posts:         Optional[float],
        account_type:      Optional[str],
        image_grade:       Optional[float],
        description_grade: Optional[float],
        timestamp:         Optional[str],
        extra_features:    Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        missing_features     = []
        filled_with_zero     = []
        unsupported_features = []

        dt             = self._parse_timestamp(timestamp)
        extra_features = extra_features or {}

        for col in self.conf_cols:
            if col == "is_business_account":
                if account_type is None or str(account_type).strip() == "":
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col in {"followers", "log_followers"}:
                if followers is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col in {"following", "log_following"}:
                if following is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col in {"num_posts", "log_num_posts"}:
                if num_posts is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col == "follow_ratio":
                if followers is None or following is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col == "posts_per_follower":
                if num_posts is None or followers is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col in self._time_features:
                if dt is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col == "image_grade":
                if image_grade is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col == "description_grade":
                if description_grade is None:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            elif col in self._surface_features:
                continue  # завжди обчислюються з тексту
            elif col.startswith("cat_"):
                if col not in extra_features:
                    missing_features.append(col)
                    filled_with_zero.append(col)
            else:
                unsupported_features.append(col)

        stage_a_usable = len(unsupported_features) == 0

        if stage_a_usable and len(filled_with_zero) == 0:
            quality = "complete"
        elif stage_a_usable:
            quality = "partial"
        else:
            quality = "unsupported"

        return {
            "usable":                   stage_a_usable,
            "quality":                  quality,
            "missing_features":         missing_features,
            # Backward-compatible field. Values are no longer filled by zero;
            # they are imputed by StandardScaler training means in _extract_confounders.
            "filled_with_zero":         [],
            "imputed_with_train_mean":  filled_with_zero,
            "unsupported_features":     unsupported_features,
        }

    # -------------------------
    # Stage A feature extraction
    # Усі формули вивірені з тренувальним кодом (Colab notebook)
    # -------------------------
    def _extract_confounders(
        self,
        clean_text:        str,
        followers:         Optional[float],
        following:         Optional[float],
        num_posts:         Optional[float],
        account_type:      Optional[str],
        image_grade:       Optional[float],
        description_grade: Optional[float],
        timestamp:         Optional[str],
        extra_features:    Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        t              = str(clean_text or "")  # вже lowercase після clean_text()
        followers_v    = max(0.0, self._safe_float(followers,  0.0))
        following_v    = max(0.0, self._safe_float(following,  0.0))
        num_posts_v    = max(0.0, self._safe_float(num_posts,  0.0))
        dt             = self._parse_timestamp(timestamp)
        extra_features = extra_features or {}

        feats: Dict[str, float] = {}

        for col in self.conf_cols:

            # --- Акаунт-метрики ---
            if col == "is_business_account":
                if self._is_missing_value(account_type):
                    feats[col] = self._train_feature_mean(col, 0.0)
                else:
                    feats[col] = 1.0 if str(account_type or "").upper() == "BUSINESS" else 0.0

            elif col == "followers":
                feats[col] = self._value_or_train_mean(col, followers, followers_v)
            elif col == "following":
                feats[col] = self._value_or_train_mean(col, following, following_v)
            elif col == "num_posts":
                feats[col] = self._value_or_train_mean(col, num_posts, num_posts_v)

            elif col == "log_followers":
                feats[col] = self._value_or_train_mean(col, followers, math.log1p(followers_v))
            elif col == "log_following":
                feats[col] = self._value_or_train_mean(col, following, math.log1p(following_v))
            elif col == "log_num_posts":
                feats[col] = self._value_or_train_mean(col, num_posts, math.log1p(num_posts_v))

            # FIX: було followers/max(1,following) — тепер відповідає тренуванню
            # Тренування: df["follow_ratio"] = df["following"] / (df["followers"] + 1)
            elif col == "follow_ratio":
                if self._is_missing_value(followers) or self._is_missing_value(following):
                    feats[col] = self._train_feature_mean(col, 0.0)
                else:
                    feats[col] = following_v / (followers_v + 1.0)

            # Тренування: df["posts_per_follower"] = df["num_posts"] / (df["followers"] + 1)
            elif col == "posts_per_follower":
                if self._is_missing_value(followers) or self._is_missing_value(num_posts):
                    feats[col] = self._train_feature_mean(col, 0.0)
                else:
                    feats[col] = num_posts_v / (followers_v + 1.0)

            # --- Interaction features (з Colab notebook) ---
            elif col == "hashtags_per_word":
                wc = float(len(t.split())) if t else 0.0
                feats[col] = t.count("#") / (wc + 1.0)

            elif col == "mentions_per_word":
                wc = float(len(t.split())) if t else 0.0
                feats[col] = t.count("@") / (wc + 1.0)

            elif col == "emoji_per_word":
                wc = float(len(t.split())) if t else 0.0
                ec = float(sum(1 for c in t if emoji_lib.is_emoji(c)))
                feats[col] = ec / (wc + 1.0)

            elif col == "exclam_per_len":
                feats[col] = float(t.count("!")) / (float(len(t)) + 1.0)

            elif col == "question_per_len":
                feats[col] = float(t.count("?")) / (float(len(t)) + 1.0)

            # --- Часові ---
            elif col == "month":
                feats[col] = float(dt.month) if dt is not None else self._train_feature_mean(col, 0.0)
            elif col == "is_weekend":
                feats[col] = 1.0 if (dt is not None and dt.weekday() >= 5) else self._train_feature_mean(col, 0.0)
            elif col == "dow_sin":
                feats[col] = float(math.sin(2 * math.pi * dt.weekday() / 7.0)) if dt is not None else self._train_feature_mean(col, 0.0)
            elif col == "dow_cos":
                feats[col] = float(math.cos(2 * math.pi * dt.weekday() / 7.0)) if dt is not None else self._train_feature_mean(col, 0.0)

            # --- Якість ---
            elif col == "image_grade":
                feats[col] = self._value_or_train_mean(col, image_grade, self._safe_float(image_grade, 0.0))
            elif col == "description_grade":
                feats[col] = self._value_or_train_mean(col, description_grade, self._safe_float(description_grade, 0.0))

            # --- Surface-ознаки тексту ---
            # FIX: text.count("#") — відповідає тренуванню (було re.findall)
            elif col == "text_len":
                feats[col] = float(len(t))
            elif col == "word_count":
                feats[col] = float(len(t.split()))
            elif col == "hashtag_count":
                # FIX: тренування використовувало text.count("#"), не findall
                feats[col] = float(t.count("#"))
            elif col == "mention_count":
                # FIX: тренування використовувало text.count("@"), не findall
                feats[col] = float(t.count("@"))
            elif col == "exclamation_count":
                feats[col] = float(t.count("!"))
            elif col == "question_count":
                feats[col] = float(t.count("?"))
            elif col == "emoji_count":
                # FIX: тренування використовувало emoji.is_emoji() з бібліотеки emoji
                # (покриває BMP + supplementary, на відміну від regex [\U00010000-\U0010ffff])
                feats[col] = float(sum(1 for c in t if emoji_lib.is_emoji(c)))

            # --- Категорії ---
            elif col.startswith("cat_"):
                feats[col] = self._value_or_train_mean(col, extra_features.get(col), self._safe_float(extra_features.get(col), 0.0))

            else:
                raise ValueError(f"Unsupported confounder at inference time: {col}")

        return feats

    # -------------------------
    # Attention-based token saliency
    # -------------------------
    @staticmethod
    def _token_importance_from_attn(last_layer_attn: torch.Tensor) -> np.ndarray:
        cls_to_tokens = last_layer_attn[:, :, 0, :]
        imp = cls_to_tokens.mean(dim=1)
        return imp.squeeze(0).detach().cpu().numpy()

    def _clean_word(self, w: str) -> str:
        w = w.strip()
        w = re.sub(r"^[#@]+", "", w)
        w = re.sub(r"[^A-Za-z0-9\-']+", "", w)
        return w

    def _attention_terms(
        self,
        input_ids_1d: torch.Tensor,
        token_scores:  np.ndarray,
        top_k:         int = 6
    ) -> List[Dict[str, int]]:
        toks = self.tokenizer.convert_ids_to_tokens(input_ids_1d.tolist())
        words, scores = [], []
        cur, cur_s = "", 0.0

        for tok, sc in zip(toks, token_scores):
            if tok in ("<s>", "</s>", "<pad>"):
                continue
            if tok.startswith("\u0120"):  # Ġ
                if cur:
                    words.append(cur)
                    scores.append(cur_s)
                cur   = tok[1:]
                cur_s = float(sc)
            else:
                cur   += tok
                cur_s  = max(cur_s, float(sc))

        if cur:
            words.append(cur)
            scores.append(cur_s)

        agg: Dict[str, float] = {}
        for w, s in zip(words, scores):
            w  = self._clean_word(w)
            wl = w.lower()
            if len(w) < 3 or wl in self._stop or wl.isdigit() or len(w) > 28:
                continue
            agg[wl] = max(agg.get(wl, 0.0), float(s))

        if not agg:
            return []

        items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        mx    = max(v for _, v in items) if items else 1.0
        return [{"term": w, "score": int(round((s / mx) * 100.0))} for w, s in items]

    # -------------------------
    # Auxiliary linguistic analysis
    # Примітка: використовує raw_text (до clean_text), щоб зберегти
    # стилістичні сигнали (емодзі, пунктуація) для рекомендацій.
    # Ці функції НЕ впливають на модельні передбачення.
    # -------------------------
    def _linguistic_profile(self, text: str) -> Dict[str, Any]:
        t   = str(text or "")
        t_l = t.lower()

        flesch = self._safe_float(textstat.flesch_reading_ease(t),            0.0)
        fk     = self._safe_float(textstat.flesch_kincaid_grade(t),           0.0)
        smog   = self._safe_float(textstat.smog_index(t),                     0.0)
        ari    = self._safe_float(textstat.automated_readability_index(t),    0.0)

        words    = re.findall(r"[A-Za-z']+", t)
        wc       = len(words)
        uniq     = len(set(w.lower() for w in words)) if wc else 0
        ttr      = float(uniq) / float(max(1, wc))

        sent_cnt     = max(1, len(re.findall(r"[.!?]+", t)))
        avg_sent_len = float(wc) / float(sent_cnt)

        hashtag_cnt = len(re.findall(r"#\w+", t))
        mention_cnt = len(re.findall(r"@\w+", t))
        q_cnt       = t.count("?")
        ex_cnt      = t.count("!")
        emoji_cnt   = sum(1 for c in t if emoji_lib.is_emoji(c))

        cta = re.findall(r"\b(click|link|comment|share|save|dm|buy|join|try|drop|check)\b", t_l)
        you = re.findall(r"\b(you|your)\b", t_l)

        blob = TextBlob(t)
        pol  = self._safe_float(blob.sentiment.polarity, 0.0)
        pos  = max(0.0, pol)  * 100.0
        neg  = max(0.0, -pol) * 100.0
        neu  = max(0.0, 100.0 - pos - neg)

        return {
            "readability": {
                "flesch":  round(flesch, 1),
                "fk_grade": round(fk,   2),
                "smog":    round(smog,   2),
                "ari":     round(ari,    2),
            },
            "style": {
                "word_count":           int(wc),
                "sentence_count":       int(sent_cnt),
                "avg_sentence_len":     round(avg_sent_len, 2),
                "lexical_diversity_ttr": round(ttr, 3),
                "hashtag_count":        int(hashtag_cnt),
                "mention_count":        int(mention_cnt),
                "question_count":       int(q_cnt),
                "exclamation_count":    int(ex_cnt),
                "emoji_count":          int(emoji_cnt),
                "cta_hits":             int(len(cta)),
                "direct_address_hits":  int(len(you)),
            },
            "sentiment": {
                "polarity": round(pol, 3),
                "pos_pct":  round(pos, 1),
                "neg_pct":  round(neg, 1),
                "neu_pct":  round(neu, 1),
            },
            "signals": {
                "cta":            bool(len(cta) > 0),
                "direct_address": bool(len(you) > 0),
                "question":       bool(q_cnt > 0),
                "hashtags_ok":    bool(2 <= hashtag_cnt <= 8),
            },
        }

    def _recommendations(self, profile: Dict[str, Any], audience: str = "generic") -> Dict[str, List[str]]:
        reasons, improvements = [], []
        r   = profile["readability"]
        s   = profile["style"]
        pol = profile["sentiment"]["polarity"]

        if r["flesch"] >= 60:
            reasons.append("Readability is within an accessible range.")
        else:
            improvements.append("Increase readability by shortening sentences.")

        if s["avg_sentence_len"] > 22:
            improvements.append("Reduce average sentence length.")

        if s["cta_hits"] > 0:
            reasons.append("Call-to-action markers are present.")
        else:
            improvements.append("Add an explicit call-to-action.")

        if s["hashtag_count"] == 0:
            improvements.append("Add 2–5 topic hashtags.")
        elif s["hashtag_count"] > 12:
            improvements.append("Reduce hashtag count.")

        if pol > 0.25:
            reasons.append("Positive framing signal is detected.")

        aud = (audience or "generic").lower()
        if aud in ("b2b", "professional", "tech"):
            if r["fk_grade"] < 7:
                improvements.append("Increase informational density for B2B.")
            if s["word_count"] < 35:
                improvements.append("Increase completeness.")
        if aud in ("b2c", "general"):
            if r["fk_grade"] > 10:
                improvements.append("Reduce complexity for a broad audience.")

        return {"reasons": reasons[:3], "improvements": improvements[:3]}

    def _language_quality_score(self, profile: Dict[str, Any]) -> float:
        r, s         = profile["readability"], profile["style"]
        flesch_norm  = np.clip(r["flesch"] / 100.0, 0.0, 1.0)
        sent_penalty = np.clip((s["avg_sentence_len"] - 15.0) / 30.0, 0.0, 1.0)
        diversity    = np.clip(s["lexical_diversity_ttr"], 0.0, 1.0)
        score        = 0.5 * flesch_norm + 0.3 * diversity + 0.2 * (1.0 - sent_penalty)
        return round(float(score * 100.0), 1)

    def _audience_fit_score(self, profile: Dict[str, Any], audience: str) -> float:
        s, r = profile["style"], profile["readability"]
        aud  = (audience or "generic").lower()
        base = 0.5

        if aud == "b2b":
            if r["fk_grade"] >= 8:   base += 0.20
            if s["word_count"] > 40: base += 0.15
            if s["cta_hits"] > 0:    base += 0.10
        elif aud in ("general", "b2c"):
            if r["fk_grade"] <= 9:   base += 0.20
            if s["emoji_count"] > 0: base += 0.10

        return round(float(np.clip(base, 0.0, 1.0) * 100.0), 1)

    def _engagement_potential(self, profile: Dict[str, Any]) -> float:
        s    = profile["style"]
        base = 0.3
        if s["cta_hits"] > 0:               base += 0.20
        if s["question_count"] > 0:         base += 0.15
        if 2 <= s["hashtag_count"] <= 8:    base += 0.15
        if s["direct_address_hits"] > 0:    base += 0.10
        return round(float(np.clip(base, 0.0, 1.0) * 100.0), 1)

    def _calibrated_index(self, pred_log: float) -> float:
        p05   = float(self.index_cal.get("p05", 0.586))
        p95   = float(self.index_cal.get("p95", 3.581))
        denom = (p95 - p05) if abs(p95 - p05) > 1e-9 else 1.0
        z     = (float(pred_log) - p05) / denom
        return round(float(np.clip(z * 100.0, 0.0, 100.0)), 1)

    def _calibrated_text_effect_score(self, text_effect: float) -> float:
        p05   = float(self.text_effect_cal.get("p05", -0.5))
        p95   = float(self.text_effect_cal.get("p95",  0.5))
        denom = (p95 - p05) if abs(p95 - p05) > 1e-9 else 1.0
        z     = (float(text_effect) - p05) / denom
        return round(float(np.clip(z * 100.0, 0.0, 100.0)), 1)

    # -------------------------
    # Public API
    # -------------------------
    def predict(
        self,
        text:              str,
        followers:         Optional[float] = None,
        following:         Optional[float] = None,
        num_posts:         Optional[float] = None,
        account_type:      Optional[str]   = None,
        audience:          str             = "generic",
        image_grade:       Optional[float] = None,
        description_grade: Optional[float] = None,
        timestamp:         Optional[str]   = None,
        extra_features:    Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_text   = str(text or "")
        clean_text = self.clean_text(raw_text)  # lowercase + URL-видалення

        if not clean_text:
            return {"ok": False, "error": "EMPTY_TEXT_AFTER_CLEANING"}

        stage_a_status = self._collect_stage_a_diagnostics(
            followers=followers, following=following, num_posts=num_posts,
            account_type=account_type, image_grade=image_grade,
            description_grade=description_grade, timestamp=timestamp,
            extra_features=extra_features,
        )

        # Stage B — text-only
        enc = self.tokenizer(
            clean_text,
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.inference_mode():
            pred_b, last_attn = self.model_B(input_ids, attention_mask)

        text_residual_component = float(pred_b.item())

        # Stage A — confounders
        baseline_component       = None
        baseline_component_raw   = None
        predicted_engagement_log = None
        predicted_engagement_log_raw = None
        prediction_mode          = "text_only_analysis"
        stage_a_feature_debug    = None

        if stage_a_status["usable"]:
            feats = self._extract_confounders(
                clean_text=clean_text,
                followers=followers, following=following, num_posts=num_posts,
                account_type=account_type, image_grade=image_grade,
                description_grade=description_grade, timestamp=timestamp,
                extra_features=extra_features,
            )

            x_arr = np.array([[feats[c] for c in self.conf_cols]], dtype=np.float32)
            guard = self._scale_and_guard_stage_a_features(x_arr)
            feats_scaled = guard["scaled"]
            feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32, device=self.device)

            with torch.inference_mode():
                pred_a = self.model_A(feats_tensor)

            baseline_component_raw = float(pred_a.item())
            baseline_component = self._clamp_engagement_log(baseline_component_raw)
            predicted_engagement_log_raw = baseline_component + text_residual_component
            predicted_engagement_log = self._clamp_engagement_log(predicted_engagement_log_raw)
            prediction_mode = "full_prediction"

            bounds = self._engagement_log_bounds()
            stage_a_feature_debug = {
                "max_abs_z_raw": guard["max_abs_z_raw"],
                "z_clipped": guard["clipped"],
                "z_clip_limit": guard["z_clip_limit"],
                "ood_features": guard["ood_features"],
                "output_bounds": bounds,
                "baseline_component_raw": round(float(baseline_component_raw), 4),
                "baseline_component_clamped": round(float(baseline_component), 4),
            }

        attention_terms = self._attention_terms(
            enc["input_ids"][0].detach().cpu(),
            self._token_importance_from_attn(last_attn),
            top_k=6
        )

        # Auxiliary analysis використовує raw_text (до lowercase) для збереження
        # стилістичних сигналів (регістр, пунктуація) у рекомендаціях
        profile         = self._linguistic_profile(raw_text)
        recommendations = self._recommendations(profile, audience=audience)

        return {
            "ok":              True,
            "prediction_mode": prediction_mode,

            "stage_a_status": stage_a_status,

            "model_output": {
                "predicted_engagement_log": (
                    round(float(predicted_engagement_log), 4)
                    if predicted_engagement_log is not None else None
                ),
                "baseline_component": (
                    round(float(baseline_component), 4)
                    if baseline_component is not None else None
                ),
                "text_residual_component": round(float(text_residual_component), 4),
            },

            "relative_scores": {
                "engagement_relative_score": (
                    self._calibrated_index(predicted_engagement_log)
                    if predicted_engagement_log is not None else None
                ),
                "text_effect_relative_score": self._calibrated_text_effect_score(
                    text_residual_component
                ),
            },

            "attention_analysis": {
                "terms": attention_terms,
                "note":  "Attention-derived token saliency is heuristic and not a causal explanation."
            },

            "auxiliary_analysis": {
                "language_quality_score":    self._language_quality_score(profile),
                "audience_fit_score":        self._audience_fit_score(profile, audience),
                "engagement_potential_score": self._engagement_potential(profile),
                "recommendations":           recommendations,
                "profile":                   profile,
            },

            "debug": {
                "used_clean_text":          clean_text,
                "index_calibration":        dict(self.index_cal),
                "text_effect_calibration":  dict(self.text_effect_cal),
                "conf_cols":                list(self.conf_cols),
                "stage_a_feature_diagnostics": stage_a_feature_debug,
                "raw_model_output": {
                    "baseline_component_raw": (
                        round(float(baseline_component_raw), 4)
                        if baseline_component_raw is not None else None
                    ),
                    "predicted_engagement_log_raw": (
                        round(float(predicted_engagement_log_raw), 4)
                        if predicted_engagement_log_raw is not None else None
                    ),
                },
            }
        }