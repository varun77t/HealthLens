"""Phase 11d tests: saving is explicit, scoped to one account, and honestly recorded.

The three that matter most:

* :func:`test_a_client_supplied_probability_is_ignored` — the server re-runs the model and
  stores its own answer. Without this the history is a table of numbers the browser
  asserted, shown back to someone as their own health record.
* :func:`test_one_user_cannot_read_another_users_analysis` — and it 404s rather than 403s,
  so the endpoint is not an oracle for which ids exist.
* :func:`test_a_saved_analysis_pins_the_threshold_and_model_version` — a retrain moves the
  operating threshold, and a stored probability rendered against a later one would reverse
  whether the model flagged the case.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.main import app
from backend.schemas.features import example_payload
from backend.tables import Analysis, User
from tests.conftest import PASSWORD, register, unique_email

DISEASES = ("heart", "kidney", "diabetes")


def _save(c: TestClient, disease: str = "heart", **overrides) -> dict:
    body = {
        "disease": disease,
        "features": example_payload(disease),
        "source_kind": "manual",
        **overrides,
    }
    r = c.post("/analyses", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# --- saving --------------------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_saving_stores_the_result_the_server_produced(client, disease):
    live = client.post(f"/predict/{disease}", json=example_payload(disease)).json()
    saved = _save(client, disease)

    assert saved["probability"] == pytest.approx(live["probability"])
    assert saved["flagged"] == live["flagged"]
    assert saved["threshold"] == pytest.approx(live["threshold"])
    assert saved["band_label"] == live["risk_band"]["label"]
    assert saved["model_version"] == live["model_version"]
    assert saved["explanation"]["contributions"], "the explanation must be stored, not re-derived later"


def test_a_client_supplied_probability_is_ignored(client):
    """The save request has no field for a result, and smuggling one in changes nothing.

    `extra="forbid"` on the feature model means a stray key is rejected outright rather than
    silently dropped, which is the stronger behaviour: a caller that thought it was setting
    the probability is told it was not.
    """
    live = client.post("/predict/heart", json=example_payload("heart")).json()

    r = client.post(
        "/analyses",
        json={
            "disease": "heart",
            "features": {**example_payload("heart"), "probability": 0.01},
            "source_kind": "manual",
        },
    )
    assert r.status_code == 422

    # And a top-level one is simply not part of the request model.
    saved = _save(client, "heart", label="attempted override")
    assert saved["probability"] == pytest.approx(live["probability"])
    assert saved["probability"] > 0.01


def test_a_saved_analysis_pins_the_threshold_and_model_version(client, db):
    saved = _save(client, "diabetes")
    row = db.execute(select(Analysis).where(Analysis.id == saved["id"])).scalar_one()

    live = client.get("/models").json()
    card = next(c for c in live if c["disease"] == "diabetes")

    assert row.threshold == pytest.approx(card["operating_threshold"])
    assert row.model_version
    assert saved["model_is_current"] is True
    # The band is stored with its own boundaries, so it can be rendered without the model.
    assert row.band_lower <= row.probability <= row.band_upper
    assert (row.probability >= row.threshold) == row.flagged


def test_a_stale_model_version_is_reported_not_re_scored(client, db):
    """Simulates a retrain: the stored result stays, and the entry says the model moved."""
    saved = _save(client, "heart")
    row = db.execute(select(Analysis).where(Analysis.id == saved["id"])).scalar_one()
    original_probability = row.probability
    row.model_version = "0.0.0-earlier"
    db.commit()

    detail = client.get(f"/analyses/{saved['id']}").json()
    assert detail["model_is_current"] is False
    assert detail["current_model_version"] not in (None, "0.0.0-earlier")
    assert detail["probability"] == pytest.approx(original_probability), (
        "an old result must never be silently re-scored against a newer model"
    )


def test_an_empty_case_cannot_be_saved(client):
    empty = {k: None for k in example_payload("heart")}
    r = client.post("/analyses", json={"disease": "heart", "features": empty})
    assert r.status_code == 422
    assert "describes no case" in r.text


def test_an_unknown_category_cannot_be_saved(client):
    payload = {**example_payload("heart"), "cp": 99}
    r = client.post("/analyses", json={"disease": "heart", "features": payload})
    assert r.status_code == 422


def test_the_source_document_is_a_filename_and_the_document_is_not_stored(client, db):
    saved = _save(
        client, "kidney", source_kind="upload", source_document="kidney-positive-1.pdf"
    )
    row = db.execute(select(Analysis).where(Analysis.id == saved["id"])).scalar_one()
    assert row.source_document == "kidney-positive-1.pdf"
    # Nothing on the row holds document bytes: every column is a scalar or JSON of scalars.
    assert not any(isinstance(v, (bytes, bytearray)) for v in vars(row).values())


# --- listing and ownership -------------------------------------------------------------------


def test_history_lists_newest_first_and_filters_by_module(client):
    for disease in ("heart", "kidney", "heart"):
        _save(client, disease)

    page = client.get("/analyses").json()
    assert page["total"] >= 3
    stamps = [item["created_at"] for item in page["items"]]
    assert stamps == sorted(stamps, reverse=True)

    only_kidney = client.get("/analyses", params={"disease": "kidney"}).json()
    assert {i["disease"] for i in only_kidney["items"]} == {"kidney"}


def test_history_paginates(client):
    for _ in range(3):
        _save(client, "heart")
    first = client.get("/analyses", params={"limit": 2, "offset": 0}).json()
    second = client.get("/analyses", params={"limit": 2, "offset": 2}).json()
    assert len(first["items"]) == 2
    assert first["total"] == second["total"]
    assert {i["id"] for i in first["items"]}.isdisjoint({i["id"] for i in second["items"]})


def test_one_user_cannot_read_another_users_analysis(client):
    mine = _save(client, "heart")

    other = TestClient(app)
    register(other)
    assert other.get(f"/analyses/{mine['id']}").status_code == 404, (
        "404, not 403 — a 403 would confirm the id exists"
    )
    assert other.request("DELETE", f"/analyses/{mine['id']}").status_code == 404
    assert other.patch(f"/analyses/{mine['id']}", json={"label": "theirs"}).status_code == 404
    assert other.get("/analyses").json()["total"] == 0

    # And it is untouched.
    assert client.get(f"/analyses/{mine['id']}").status_code == 200


def test_saved_analyses_require_a_session(anon_client):
    for method, path in (
        ("GET", "/analyses"),
        ("POST", "/analyses"),
        ("GET", "/analyses/anything"),
        ("DELETE", "/analyses/anything"),
    ):
        assert anon_client.request(method, path, json={}).status_code == 401


# --- editing and deleting -----------------------------------------------------------------------


def test_only_the_label_can_be_changed(client):
    saved = _save(client, "heart", label="before")
    updated = client.patch(f"/analyses/{saved['id']}", json={"label": "after"}).json()

    assert updated["label"] == "after"
    # Everything about the result is unchanged: a saved analysis is a record, not a draft.
    for field in ("probability", "threshold", "band_label", "model_version", "features"):
        assert updated[field] == saved[field]


def test_deleting_a_saved_analysis_removes_it(client, db):
    saved = _save(client, "heart")
    assert client.request("DELETE", f"/analyses/{saved['id']}").status_code == 204
    assert client.get(f"/analyses/{saved['id']}").status_code == 404
    assert db.execute(select(Analysis).where(Analysis.id == saved["id"])).scalar_one_or_none() is None


def test_deleting_the_account_deletes_every_saved_analysis(client, db):
    # `client` is requested only to keep the module's lifespan open: the registry lives on
    # app.state, and entering a second TestClient context would tear it down on exit and
    # leave every later test in this module hitting a 503.
    c = TestClient(app)
    email = unique_email()
    register(c, email=email)
    _save(c, "heart")
    _save(c, "kidney")

    user = db.execute(select(User).where(User.email == email)).scalar_one()
    user_id = user.id
    assert db.execute(select(Analysis).where(Analysis.user_id == user_id)).scalars().all()

    assert c.request(
        "DELETE", "/auth/account", json={"password": PASSWORD, "confirm": "DELETE"}
    ).status_code == 204

    db.expire_all()
    assert db.execute(select(Analysis).where(Analysis.user_id == user_id)).scalars().all() == []


# --- the caveats travel with the record ------------------------------------------------------------


def test_a_saved_analysis_carries_its_disclaimer_and_warnings(client):
    saved = _save(client, "diabetes")
    assert "not be interpreted as medical diagnosis" in saved["disclaimer"]
    assert isinstance(saved["warnings"], list)
    assert saved["module"]


def test_the_list_explains_that_results_are_never_re_scored(client):
    _save(client, "heart")
    note = client.get("/analyses").json()["note"]
    assert "re-scored" in note and "model version" in note
