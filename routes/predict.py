"""Prediction and history routes blueprint."""

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    g,
)

from shared import (
    api_error as api_response_error,
    api_response,
    approved_user_required,
    csrf_required,
    current_account,
    login_required,
    record_history,
)
from gtts import gTTS
from io import BytesIO
from flask import send_file
from models.predictor import run_prediction, model_assets_ready

bp = Blueprint("predict", __name__)


@bp.route("/live")
@approved_user_required()
def live():
    account = g.account
    has_model = model_assets_ready()
    expected_sequence_length = None
    if has_model:
        try:
            from models.predictor import prediction_sequence_length

            expected_sequence_length = prediction_sequence_length()
        except Exception:
            expected_sequence_length = 20
    return render_template(
        "live.html",
        has_model=has_model,
        account=account,
        expected_sequence_length=expected_sequence_length,
    )


@bp.route("/predict", methods=["POST"])
@approved_user_required(api=True)
@csrf_required
def predict():
    account = g.account
    if not request.is_json:
        return api_response_error("Expected application/json body.", 400)

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    image_sequence = payload.get("images")

    if not image_sequence:
        return api_response_error(
            "A complete frame sequence is required for live prediction.", 400
        )
    if image_data:
        return api_response_error(
            "Single-frame prediction is disabled for the sequence model.", 400
        )

    try:
        from models.predictor import load_model_and_labels

        load_model_and_labels(app_logger=current_app.logger)
    except (FileNotFoundError, ValueError) as exc:
        return api_response_error(str(exc), 503)

    return run_prediction(account, image_sequence)


@bp.post("/history/confirm")
@approved_user_required(api=True)
@csrf_required
def confirm_history():
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    if not label:
        return api_response_error("A confirmed label is required.", 400)
    try:
        import models.predictor as predictor

        predictor.load_model_and_labels()
        labels = predictor.label_classes
    except (FileNotFoundError, ValueError):
        return api_response_error("Model assets are unavailable.", 503)
    if labels is None or label not in {str(value) for value in labels}:
        return api_response_error("Unknown label.", 400)
    if g.account["role"] == "user":
        record_history(g.account["id"], label)
    return api_response({"label": label})


@bp.post("/history/clear")
@approved_user_required(api=True)
@csrf_required
def clear_history():
    account = g.account
    if account["role"] == "admin":
        return api_response_error("Admins do not have history.", 403)
    from database import get_connection

    with get_connection() as connection:
        connection.execute(
            'DELETE FROM "UserHistory" WHERE userId = ?', (account["id"],)
        )
    if request.is_json:
        return api_response({"cleared": True})
    flash("History cleared.", "success")
    return redirect(url_for("predict.history"))


@bp.route("/history")
@approved_user_required()
def history():
    account = g.account
    if account["role"] == "admin":
        return redirect(url_for("admin.admin_dashboard"))
    from database import get_connection

    with get_connection() as connection:
        rows = connection.execute(
            'SELECT id, interpretedText, timestamp FROM "UserHistory" WHERE userId = ? ORDER BY timestamp DESC, id DESC',
            (account["id"],),
        ).fetchall()
    return render_template("history.html", account=account, history=rows)


@bp.route("/tts")
@approved_user_required(api=True)
def tts():
    text = request.args.get("text", "")
    if not text:
        return api_response_error("Missing text query parameter.", 400)

    tts_audio = gTTS(text=text, lang="en")
    mp3_buffer = BytesIO()
    tts_audio.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return send_file(mp3_buffer, mimetype="audio/mpeg")
