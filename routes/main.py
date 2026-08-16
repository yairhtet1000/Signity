"""Main routes blueprint."""

from flask import Blueprint, render_template

from shared import current_account

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    from utils import dataset_summary
    from models.predictor import model_assets_ready

    summary = dataset_summary()
    has_model = model_assets_ready(summary)
    return render_template(
        "index.html",
        summary=summary,
        has_model=has_model,
        account=current_account(),
    )


@bp.route("/dataset")
def dataset():
    return render_template("dataset.html", account=current_account())
