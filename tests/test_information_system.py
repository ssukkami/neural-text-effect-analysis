import io

import pandas as pd

from conftest import assert_finite_number, assert_status, require_key


class TestInformationSystemFunctions:
    def test_single_text_full_prediction(self, client, endpoints):
        payload = {
            "text": (
                "Just deployed my first machine learning model to production. "
                "The debugging process was incredibly tough, but optimizing those "
                "hyperparameters finally paid off. What is the most challenging bug "
                "you have crushed this week? Share your experience below. "
                "#AI #MachineLearning #DevLife"
            ),
            "followers": 1250,
            "following": 340,
            "num_posts": 112,
            "timestamp": "2026-05-10 18:30:00",
            "account_type": "personal",
            "audience": "tech",
        }

        response = client.post(endpoints["single_text"], json=payload)
        data = assert_status(response, 200)

        assert isinstance(data, dict)
        assert data.get("ok") is True

        prediction_mode = require_key(data, "prediction_mode")
        model_output = require_key(data, "model_output")
        relative_scores = require_key(data, "relative_scores")
        stage_a_status = require_key(data, "stage_a_status")

        assert prediction_mode == "full_prediction"
        assert stage_a_status["usable"] is True

        predicted_log = require_key(model_output, "predicted_engagement_log")
        baseline = require_key(model_output, "baseline_component")
        residual = require_key(model_output, "text_residual_component")

        assert_finite_number(predicted_log, "model_output.predicted_engagement_log")
        assert_finite_number(baseline, "model_output.baseline_component")
        assert_finite_number(residual, "model_output.text_residual_component")

        assert abs(predicted_log - (baseline + residual)) < 1e-3

        engagement_score = require_key(relative_scores, "engagement_relative_score")
        text_effect_score = require_key(relative_scores, "text_effect_relative_score")

        assert_finite_number(engagement_score, "relative_scores.engagement_relative_score")
        assert_finite_number(text_effect_score, "relative_scores.text_effect_relative_score")

        assert 0 <= engagement_score <= 100
        assert 0 <= text_effect_score <= 100

    def test_single_text_text_only_mode(self, client, endpoints):
        payload = {
            "text": (
                "New machine learning model is live. "
                "What debugging issue should be tested next?"
            )
        }

        response = client.post(endpoints["single_text"], json=payload)
        data = assert_status(response, 200)

        assert isinstance(data, dict)
        assert data.get("ok") is True

        prediction_mode = require_key(data, "prediction_mode")
        model_output = require_key(data, "model_output")
        relative_scores = require_key(data, "relative_scores")

        assert prediction_mode == "text_only_analysis"

        assert model_output["predicted_engagement_log"] is None
        assert model_output["baseline_component"] is None

        residual = require_key(model_output, "text_residual_component")
        text_effect = require_key(relative_scores, "text_effect_relative_score")

        assert_finite_number(residual, "model_output.text_residual_component")
        assert_finite_number(text_effect, "relative_scores.text_effect_relative_score")

        assert 0 <= text_effect <= 100

    def test_single_text_empty_text_returns_error(self, client, endpoints):
        payload = {
            "text": "",
            "followers": 1250,
            "following": 340,
            "num_posts": 112,
        }

        response = client.post(endpoints["single_text"], json=payload)
        data = assert_status(response, 400)

        assert isinstance(data, dict)
        assert data.get("ok") is False
        assert data.get("error") == "INPUT_TEXT_EMPTY"

    def test_ab_compare_returns_valid_scores_and_delta(self, client, endpoints):
        payload = {
            "textA": (
                "Complex linguistic patterns with significantly higher precision. "
                "I will be publishing a technical breakdown shortly. "
                "#MachineLearning #AI"
            ),
            "textB": (
                "Who else is battling with model training this week? "
                "Drop your biggest frustration below. "
                "#NLP #MachineLearning #CodingLife"
            ),
            "followers": 1250,
            "following": 340,
            "num_posts": 112,
            "timestamp": "2026-05-10 18:30:00",
            "account_type": "personal",
            "audience": "tech",
        }

        response = client.post(endpoints["ab_compare"], json=payload)
        data = assert_status(response, 200)

        assert isinstance(data, dict)
        assert data.get("ok") is True

        result_a = require_key(data, "A")
        result_b = require_key(data, "B")
        comparison = require_key(data, "comparison")

        assert result_a.get("ok") is True
        assert result_b.get("ok") is True

        compared_metric = require_key(comparison, "metric")
        delta = require_key(comparison, "delta_B_minus_A")

        assert compared_metric in [
            "engagement_relative_score",
            "text_effect_relative_score",
        ]

        assert_finite_number(delta, "comparison.delta_B_minus_A")

        score_a = result_a["relative_scores"][compared_metric]
        score_b = result_b["relative_scores"][compared_metric]

        assert_finite_number(score_a, f"A.relative_scores.{compared_metric}")
        assert_finite_number(score_b, f"B.relative_scores.{compared_metric}")

        assert abs(delta - round(score_b - score_a, 2)) < 1e-6

    def test_ab_compare_rejects_missing_variant(self, client, endpoints):
        payload = {
            "textA": "Only one text is provided."
        }

        response = client.post(endpoints["ab_compare"], json=payload)
        data = assert_status(response, 400)

        assert isinstance(data, dict)
        assert data.get("ok") is False
        assert data.get("error") == "NEED_TWO_TEXT_VARIANTS"

    def test_dataset_csv_processing(self, client, endpoints):
        df = pd.DataFrame(
            {
                "caption": [
                    "New AI model is live. What do you think? #AI",
                    "Weekend coding session with debugging notes.",
                    "Drop your best machine learning tip below.",
                    "Production deployment finally works after tuning.",
                ],
                "likes": [120, 85, 210, 175],
                "followers": [1250, 1250, 1250, 1250],
                "following": [340, 340, 340, 340],
                "num_posts": [112, 112, 112, 112],
                "account_type": ["personal", "personal", "personal", "personal"],
                "timestamp": [
                    "2026-05-10 18:30:00",
                    "2026-05-10 18:30:00",
                    "2026-05-10 18:30:00",
                    "2026-05-10 18:30:00",
                ],
            }
        )

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            endpoints["dataset"],
            data={
                "file": (csv_buffer, "test_instagram_data.csv"),
            },
            content_type="multipart/form-data",
        )

        data = assert_status(response, 200)

        assert isinstance(data, dict)
        assert data.get("ok") is True

        summary = require_key(data, "summary")
        detected_cols = require_key(data, "detected_cols")
        metric_summary = require_key(data, "metric_summary")
        analysis = require_key(data, "analysis")

        assert summary["total_rows"] == 4
        assert summary["valid_text_rows"] == 4
        assert summary["analyzed_rows"] == 4

        assert detected_cols["text"] == "caption"
        assert detected_cols["metric"] == "likes"
        assert detected_cols["metric_type"] == "likes"

        assert metric_summary["column"] == "likes"
        assert metric_summary["type"] == "likes"
        assert metric_summary["mean_value"] == 147.5

        assert "top_terms_by_frequency" in analysis
        assert "high_signal_terms" in analysis
        assert "sentiment_distribution" in analysis
        assert "length_distribution" in analysis
        assert "length_vs_text_effect_scatter" in analysis
        assert "mode_distribution" in analysis

        assert isinstance(analysis["top_terms_by_frequency"], list)
        assert isinstance(analysis["high_signal_terms"], list)
        assert isinstance(analysis["length_vs_text_effect_scatter"], list)

        assert analysis["mode_distribution"]["full_prediction"] == 4
        assert analysis["mode_distribution"]["text_only_analysis"] == 0

    def test_dataset_rejects_missing_file(self, client, endpoints):
        response = client.post(endpoints["dataset"])
        data = assert_status(response, 400)

        assert isinstance(data, dict)
        assert data.get("ok") is False
        assert data.get("error") == "NO_FILE_SELECTED"

    def test_dataset_without_text_column_returns_empty_analysis(self, client, endpoints):
        df = pd.DataFrame(
            {
                "wrong_column": ["x1", "x2"],
                "likes": [10, 20],
            }
        )

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            endpoints["dataset"],
            data={
                "file": (csv_buffer, "wrong_dataset.csv"),
            },
            content_type="multipart/form-data",
        )

        data = assert_status(response, 200)

        assert isinstance(data, dict)
        assert data.get("ok") is True

        assert data["summary"]["total_rows"] == 2
        assert data["summary"]["analyzed_rows"] == 0
        assert data["summary"]["valid_text_rows"] == 0

        assert data["detected_cols"]["text"] is None

        assert data["analysis"]["top_terms_by_frequency"] == []
        assert data["analysis"]["high_signal_terms"] == []
        assert data["analysis"]["length_vs_text_effect_scatter"] == []
        assert data["analysis"]["mode_distribution"]["full_prediction"] == 0
        assert data["analysis"]["mode_distribution"]["text_only_analysis"] == 0