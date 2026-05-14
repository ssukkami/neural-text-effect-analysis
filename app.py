from flask import Flask, render_template, request, jsonify
import webview
import threading
import time
import re
import pandas as pd
import numpy as np

from inference import Predictor

DEV_MODE = False
MAX_ROWS_FOR_MODEL = 1200

app = Flask(__name__)
predictor = Predictor()


# =========================
# Helpers
# =========================
def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            v = float(x)
            if not np.isfinite(v):
                return default
            return v

        s = str(x).strip()
        if s == "":
            return default

        # UI-safe parsing: supports "1250", "1 250", "1,250" and "3,14".
        s = s.replace(" ", "")
        if "," in s and "." in s:
            s = s.replace(",", "")
        elif "," in s:
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) == 3 and parts[0].replace("-", "").isdigit():
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")

        v = float(s)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_str(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _first_present(payload: dict, *keys):
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _series_get_value(series, idx, cast="float"):
    if series is None or idx not in series.index:
        return None

    val = series.loc[idx]

    if cast == "float":
        return _safe_float(val)
    if cast == "int":
        return _safe_int(val)
    if cast == "str":
        return _safe_str(val)
    return val


def _extract_extra_features_from_payload(payload: dict) -> dict:
    out = {}
    for k, v in (payload or {}).items():
        if isinstance(k, str) and k.startswith("cat_"):
            out[k] = _safe_float(v, 0.0)
    return out


def _extract_extra_features_from_row(df: pd.DataFrame, idx, predictor_obj: Predictor) -> dict:
    out = {}
    if not hasattr(predictor_obj, "conf_cols"):
        return out

    for col in predictor_obj.conf_cols:
        if isinstance(col, str) and col.startswith("cat_") and col in df.columns:
            out[col] = _safe_float(df.at[idx, col], 0.0)

    return out


# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True) or {}

        text = _safe_str(data.get("text"), "")
        if not text:
            return jsonify({"ok": False, "error": "INPUT_TEXT_EMPTY"}), 400

        result = predictor.predict(
            text=text,
            followers=_safe_float(_first_present(data, "followers", "followers_count", "follower_count")),
            following=_safe_float(_first_present(data, "following", "following_count")),
            num_posts=_safe_float(_first_present(data, "num_posts", "posts", "post_count", "posts_count")),
            account_type=_safe_str(_first_present(data, "type", "account_type", "profile_type")),
            audience=_safe_str(data.get("audience"), "generic"),
            image_grade=_safe_float(_first_present(data, "image_grade", "image_score")),
            description_grade=_safe_float(_first_present(data, "description_grade", "description_score", "profile_score")),
            timestamp=_safe_str(_first_present(data, "timestamp", "date", "created_at", "posted_at")),
            extra_features=_extract_extra_features_from_payload(data),
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ab_compare", methods=["POST"])
def ab_compare():
    try:
        data = request.get_json(force=True) or {}

        text_a = _safe_str(data.get("textA"), "")
        text_b = _safe_str(data.get("textB"), "")

        if not text_a or not text_b:
            return jsonify({"ok": False, "error": "NEED_TWO_TEXT_VARIANTS"}), 400

        common_kwargs = {
            "followers": _safe_float(_first_present(data, "followers", "followers_count", "follower_count")),
            "following": _safe_float(_first_present(data, "following", "following_count")),
            "num_posts": _safe_float(_first_present(data, "num_posts", "posts", "post_count", "posts_count")),
            "account_type": _safe_str(_first_present(data, "type", "account_type", "profile_type")),
            "audience": _safe_str(data.get("audience"), "generic"),
            "image_grade": _safe_float(_first_present(data, "image_grade", "image_score")),
            "description_grade": _safe_float(_first_present(data, "description_grade", "description_score", "profile_score")),
            "timestamp": _safe_str(_first_present(data, "timestamp", "date", "created_at", "posted_at")),
            "extra_features": _extract_extra_features_from_payload(data),
        }

        result_a = predictor.predict(text=text_a, **common_kwargs)
        result_b = predictor.predict(text=text_b, **common_kwargs)

        mode_a = result_a.get("prediction_mode")
        mode_b = result_b.get("prediction_mode")

        if mode_a == "full_prediction" and mode_b == "full_prediction":
            score_a = result_a["relative_scores"]["engagement_relative_score"]
            score_b = result_b["relative_scores"]["engagement_relative_score"]
            compared_metric = "engagement_relative_score"
        else:
            score_a = result_a["relative_scores"]["text_effect_relative_score"]
            score_b = result_b["relative_scores"]["text_effect_relative_score"]
            compared_metric = "text_effect_relative_score"

        delta = None
        if score_a is not None and score_b is not None:
            delta = round(float(score_b) - float(score_a), 2)

        return jsonify({
            "ok": True,
            "A": result_a,
            "B": result_b,
            "comparison": {
                "metric": compared_metric,
                "delta_B_minus_A": delta,
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# Dataset column detection
# =========================
COLUMN_SYNONYMS = {
    "text": [
        "caption", "text", "post_text", "post text",
        "content", "description", "body", "post", "tweet",
        "message", "tweet_text", "post_content", "post_caption",
        "caption_text", "post_body", "article", "comment_text",
        "server_post", "server post", "desc", "txt", "msg",
    ],
    "likes": [
        "likes", "like_count", "likes_count", "num_likes", "total_likes",
        "edge_media_preview_like", "likes_reactions",
        "favorites", "favorite_count", "favourites", "favourite_count",
        "hearts", "reactions", "reaction_count",
    ],
    "comments": [
        "comments", "comment_count", "comments_count", "num_comments",
        "total_comments", "reply_count", "replies", "edge_media_to_comment_count",
    ],
    "shares": [
        "shares", "share_count", "shares_retweets",
        "retweets", "retweet_count", "reposts", "repost_count", "reshares",
    ],
    "engagement": [
        "user_engagement", "engagement", "engagement_rate", "engagements",
        "total_engagement", "interactions", "interaction_count",
        "user_interactions", "user interactions", "engage",
    ],
    "views": [
        "views", "view_count", "impressions", "impression_count", "reach",
        "video_views", "play_count",
    ],
    "followers": [
        "followers", "user_followers", "user followers",
        "followers_count", "follower_count", "num_followers",
        "subscriber_count", "subscribers", "follower",
    ],
    "following": [
        "following", "user_following", "user following",
        "following_count", "followings", "friends_count", "follow",
    ],
    "num_posts": [
        "num_posts", "posts_count", "post_count", "total_posts",
        "media_count", "statuses_count", "posts",
    ],
    "account_type": [
        "is_business", "account_type", "business_account", "profile_type",
        "account_verification", "account verification", "business", "type",
    ],
    "timestamp": [
        "created_at", "taken_at_timestamp", "post_timestamp", "post timestamp",
        "timestamp", "date", "posted_at", "created", "published_at",
    ],
    "image_grade": ["image_grade", "img_grade", "photo_grade", "image_score"],
    "description_grade": ["description_grade", "desc_grade", "bio_grade", "profile_score"],
}


def detect_column(df: pd.DataFrame, field: str):
    synonyms = COLUMN_SYNONYMS.get(field, [])
    cols_lower = {c: c.lower().replace(" ", "_").replace("-", "_") for c in df.columns}

    for syn in synonyms:
        syn_l = syn.lower().replace(" ", "_").replace("-", "_")
        for orig, norm in cols_lower.items():
            if norm == syn_l:
                return orig

    for syn in synonyms:
        syn_l = syn.lower().replace(" ", "_").replace("-", "_")
        for orig, norm in cols_lower.items():
            if norm.startswith(syn_l):
                return orig

    for syn in synonyms:
        syn_l = syn.lower().replace(" ", "_").replace("-", "_")
        for orig, norm in cols_lower.items():
            if syn_l in norm:
                return orig

    return None


def detect_text_column_by_content(df: pd.DataFrame):
    skip_patterns = (
        "id", "url", "link", "date", "time", "stamp", "created", "taken",
        "img", "image", "photo", "media", "location", "loc", "place",
        "username", "user_name", "owner", "shortcode", "code", "hashtag",
        "tag", "mention", "platform", "privacy", "language", "server",
        "index", "rank"
    )

    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return None

    scores = {}
    for col in obj_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        sample = series.head(200)
        mean_len = sample.str.len().mean()
        median_len = sample.str.len().median()
        has_space = sample.str.contains(r" ").mean()
        has_punct = sample.str.contains(r"[.!?,;:]").mean()
        is_url = sample.str.contains(r"^https?://|^www\.", regex=True).mean()
        is_numeric = sample.str.match(r"^\s*[\d.,\-]+\s*$").mean()
        is_short = (sample.str.len() < 5).mean()

        score = 0.0
        if mean_len > 40:
            score += 3
        elif mean_len > 15:
            score += 1

        if median_len > 20:
            score += 2
        if has_space > 0.30:
            score += 2
        if has_punct > 0.10:
            score += 1
        if (is_url + is_numeric + is_short) > 0.40:
            score -= 5

        col_l = col.lower().replace(" ", "_").replace("-", "_")
        if any(p in col_l for p in skip_patterns):
            score -= 3

        scores[col] = score

    if not scores:
        return None

    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else None


def resolve_metric_column(df: pd.DataFrame):
    for field in ("engagement", "likes", "comments", "shares", "views"):
        col = detect_column(df, field)
        if col:
            return col, field
    return None, None


# =========================
# Dataset analytics
# =========================
@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"ok": False, "error": "NO_FILE_SELECTED"}), 400

        file = request.files["file"]

        try:
            df = pd.read_csv(file, on_bad_lines="skip")
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1", on_bad_lines="skip")

        total_rows = len(df)
        if total_rows == 0:
            return jsonify({"ok": False, "error": "EMPTY_FILE"}), 400

        text_col = detect_column(df, "text") or detect_text_column_by_content(df)
        followers_col = detect_column(df, "followers")
        following_col = detect_column(df, "following")
        posts_col = detect_column(df, "num_posts")
        type_col = detect_column(df, "account_type")
        timestamp_col = detect_column(df, "timestamp")
        image_grade_col = detect_column(df, "image_grade")
        description_grade_col = detect_column(df, "description_grade")

        metric_col, metric_type = resolve_metric_column(df)

        cat_cols_present = [
            c for c in df.columns
            if isinstance(c, str) and c.startswith("cat_") and c in predictor.conf_cols
        ]

        detected_cols = {
            "text": text_col,
            "metric": metric_col,
            "metric_type": metric_type,
            "followers": followers_col,
            "following": following_col,
            "num_posts": posts_col,
            "account_type": type_col,
            "timestamp": timestamp_col,
            "image_grade": image_grade_col,
            "description_grade": description_grade_col,
            "cat_features": cat_cols_present,
        }

        if not text_col:
            return jsonify({
                "ok": True,
                "summary": {
                    "total_rows": total_rows,
                    "analyzed_rows": 0,
                    "valid_text_rows": 0,
                    "sampling": {
                        "used_sample": False,
                        "sample_size": 0,
                        "sample_limit": MAX_ROWS_FOR_MODEL,
                    }
                },
                "detected_cols": detected_cols,
                "metric_summary": {
                    "column": metric_col,
                    "type": metric_type,
                    "mean_value": None,
                },
                "analysis": {
                    "top_terms_by_frequency": [],
                    "high_signal_terms": [],
                    "sentiment_distribution": {"positive": 0, "neutral": 100, "negative": 0},
                    "length_distribution": {
                        "labels": ["0-20", "21-50", "51-100", "101-150", "151+"],
                        "values": [0, 0, 0, 0, 0]
                    },
                    "length_vs_text_effect_scatter": [],
                    "mode_distribution": {"full_prediction": 0, "text_only_analysis": 0},
                }
            })

        use_sample = total_rows > MAX_ROWS_FOR_MODEL
        sample_df = df.sample(MAX_ROWS_FOR_MODEL, random_state=42) if use_sample else df.copy()

        followers_s = pd.to_numeric(sample_df[followers_col], errors="coerce") if followers_col else None
        following_s = pd.to_numeric(sample_df[following_col], errors="coerce") if following_col else None
        posts_s = pd.to_numeric(sample_df[posts_col], errors="coerce") if posts_col else None
        type_s = sample_df[type_col].astype(str) if type_col else None
        timestamp_s = sample_df[timestamp_col].astype(str) if timestamp_col else None
        image_grade_s = pd.to_numeric(sample_df[image_grade_col], errors="coerce") if image_grade_col else None
        description_grade_s = pd.to_numeric(sample_df[description_grade_col], errors="coerce") if description_grade_col else None

        metric_mean = None
        if metric_col:
            metric_series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
            if len(metric_series) > 0:
                metric_mean = float(metric_series.mean())

        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "your", "you", "are", "was", "were", "will", "have", "has", "had",
            "not", "but", "can", "our", "their", "they", "them", "its", "it's",
            "a", "an", "to", "of", "in", "on", "is", "it", "we", "as", "be",
            "by", "at", "or", "so", "if", "about", "out", "like", "all", "one",
            "what", "when", "who", "how", "more", "time", "people", "would",
            "could", "should", "which", "some", "there", "then", "than", "only",
            "also", "very", "even", "through", "latest", "much", "many", "other",
            "such", "most", "these", "those", "over", "under"
        }

        word_freq = {}
        word_signal = {}
        scatter = []
        pos, neu, neg = 0, 0, 0
        valid_text_rows = 0
        analyzed_rows = 0
        full_prediction_rows = 0
        text_only_rows = 0
        text_lengths = []

        text_indices = sample_df[text_col].dropna().index.tolist()

        for idx in text_indices:
            raw_text = str(sample_df.at[idx, text_col]).strip()
            if not raw_text:
                continue

            clean = predictor.clean_text(raw_text)
            if len(clean) < 2:
                continue

            valid_text_rows += 1
            text_lengths.append(len(clean))

            extra_features = _extract_extra_features_from_row(sample_df, idx, predictor)

            result = predictor.predict(
                text=raw_text,
                followers=_series_get_value(followers_s, idx, cast="float"),
                following=_series_get_value(following_s, idx, cast="float"),
                num_posts=_series_get_value(posts_s, idx, cast="float"),
                account_type=_series_get_value(type_s, idx, cast="str"),
                image_grade=_series_get_value(image_grade_s, idx, cast="float"),
                description_grade=_series_get_value(description_grade_s, idx, cast="float"),
                timestamp=_series_get_value(timestamp_s, idx, cast="str"),
                audience="generic",
                extra_features=extra_features,
            )

            if not result.get("ok"):
                continue

            analyzed_rows += 1
            mode = result.get("prediction_mode", "text_only_analysis")
            if mode == "full_prediction":
                full_prediction_rows += 1
            else:
                text_only_rows += 1

            profile = result.get("auxiliary_analysis", {}).get("profile", {})
            sentiment = profile.get("sentiment", {})
            pol = float(sentiment.get("polarity", 0.0))
            if pol > 0.05:
                pos += 1
            elif pol < -0.05:
                neg += 1
            else:
                neu += 1

            text_effect = float(result["model_output"]["text_residual_component"])
            text_effect_score = result["relative_scores"]["text_effect_relative_score"]

            words_in_text = re.findall(r"[A-Za-z']{4,}", clean.lower())
            uniq_words = {w for w in words_in_text if w not in stop_words}

            for w in uniq_words:
                word_freq[w] = word_freq.get(w, 0) + 1
                sig_sum, sig_count = word_signal.get(w, (0.0, 0))
                word_signal[w] = (sig_sum + text_effect, sig_count + 1)

            scatter.append({
                "x": int(len(words_in_text)),
                "y": float(text_effect_score),
            })

        total_sent = pos + neu + neg
        sentiment_dist = {
            "positive": round((pos / total_sent) * 100) if total_sent > 0 else 0,
            "neutral": round((neu / total_sent) * 100) if total_sent > 0 else 100,
            "negative": round((neg / total_sent) * 100) if total_sent > 0 else 0,
        }

        min_occurrences = max(2, int(max(1, analyzed_rows) * 0.01))
        weighted_terms = []
        for term, freq in word_freq.items():
            sig_sum, sig_count = word_signal.get(term, (0.0, 0))
            if freq < min_occurrences or sig_count == 0:
                continue
            mean_signal = sig_sum / float(sig_count)
            weighted_terms.append((term, freq, mean_signal, abs(mean_signal) * np.log1p(freq)))

        top_freq = sorted(weighted_terms, key=lambda x: x[1], reverse=True)[:8]
        top_signal = sorted(weighted_terms, key=lambda x: x[3], reverse=True)[:8]

        length_labels = ["0-20", "21-50", "51-100", "101-150", "151+"]
        length_values = [0, 0, 0, 0, 0]
        for ln in text_lengths:
            if ln <= 20:
                length_values[0] += 1
            elif ln <= 50:
                length_values[1] += 1
            elif ln <= 100:
                length_values[2] += 1
            elif ln <= 150:
                length_values[3] += 1
            else:
                length_values[4] += 1

        return jsonify({
            "ok": True,
            "summary": {
                "total_rows": total_rows,
                "valid_text_rows": valid_text_rows,
                "analyzed_rows": analyzed_rows,
                "sampling": {
                    "used_sample": use_sample,
                    "sample_size": int(len(sample_df)),
                    "sample_limit": MAX_ROWS_FOR_MODEL,
                }
            },
            "detected_cols": detected_cols,
            "metric_summary": {
                "column": metric_col,
                "type": metric_type,
                "mean_value": round(metric_mean, 4) if metric_mean is not None and not np.isnan(metric_mean) else None,
            },
            "analysis": {
                "top_terms_by_frequency": [
                    {"term": term, "count": int(freq)}
                    for term, freq, _, _ in top_freq
                ],
                "high_signal_terms": [
                    {
                        "term": term,
                        "count": int(freq),
                        "mean_text_signal": round(float(mean_signal), 4),
                        "weighted_signal": round(float(weighted_score), 4),
                    }
                    for term, freq, mean_signal, weighted_score in top_signal
                ],
                "sentiment_distribution": sentiment_dist,
                "length_distribution": {
                    "labels": length_labels,
                    "values": length_values,
                },
                "length_vs_text_effect_scatter": scatter[:300],
                "mode_distribution": {
                    "full_prediction": int(full_prediction_rows),
                    "text_only_analysis": int(text_only_rows),
                },
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def start_server():
    app.run(port=5000, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    if DEV_MODE:
        app.run(debug=True, port=5000)
    else:
        th = threading.Thread(target=start_server, daemon=True)
        th.start()
        time.sleep(1.5)
        webview.create_window(
            "NeuroInfluence AI",
            "http://127.0.0.1:5000",
            width=1280,
            height=800,
            background_color="#ffffff"
        )
        webview.start()