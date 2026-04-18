import os
import json
import logging
import urllib.request
import urllib.parse
from base64 import b64encode
from fastapi import APIRouter, HTTPException
from ..database import run_query, WAREHOUSE_ID
from ..scoring import get_loaded_models

logger = logging.getLogger(__name__)

INSTANCE_URL   = "https://dbc-bc7909d8-379d.cloud.databricks.com"
WORKSPACE_ID   = "7474645411427680"
DASHBOARD_ID   = "01f13b0775f71d64a0e4798da5bd4aa3"

router = APIRouter(tags=["dashboard"])


@router.get("/api/dashboard/psu")
async def psu_summary():
    return run_query("""
        SELECT
            PSU_ID,
            COUNT(*)                                                                    AS total,
            SUM(CASE WHEN priority_level = 'High Priority' THEN 1 ELSE 0 END)          AS high,
            SUM(CASE WHEN priority_level = 'Routine Care'  THEN 1 ELSE 0 END)          AS medium,
            SUM(CASE WHEN priority_level = 'Medically Fit' THEN 1 ELSE 0 END)          AS low,
            ROUND(AVG(risk_complication), 3)                                            AS avg_risk
        FROM workspace.suraksha.patients
        WHERE is_deleted = false
        GROUP BY PSU_ID
        ORDER BY high DESC, avg_risk DESC
        LIMIT 20
    """)


def _http(url: str, *, method="GET", headers=None, body=None) -> dict:
    data = body.encode() if isinstance(body, str) else body
    req  = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


@router.get("/api/dashboard/embed-token")
async def embed_token():
    """
    3-step scoped token exchange for AI/BI dashboard embedding.
    Requires env vars: DATABRICKS_SP_CLIENT_ID, DATABRICKS_SP_CLIENT_SECRET
    Optional:          DASHBOARD_EXTERNAL_VIEWER_ID, DASHBOARD_EXTERNAL_VALUE
    """
    sp_id     = os.environ.get("DATABRICKS_SP_CLIENT_ID")
    sp_secret = os.environ.get("DATABRICKS_SP_CLIENT_SECRET")
    if not sp_id or not sp_secret:
        raise HTTPException(status_code=500, detail="Service principal env vars not set")

    ext_viewer = os.environ.get("DASHBOARD_EXTERNAL_VIEWER_ID", "suraksha-anm")
    ext_value  = os.environ.get("DASHBOARD_EXTERNAL_VALUE",     "suraksha-app")
    basic_auth = b64encode(f"{sp_id}:{sp_secret}".encode()).decode()

    # Step 1 — all-apis OIDC token
    step1 = _http(
        f"{INSTANCE_URL}/oidc/v1/token",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {basic_auth}"},
        body=urllib.parse.urlencode({"grant_type": "client_credentials", "scope": "all-apis"}),
    )
    oidc_token = step1["access_token"]

    # Step 2 — token info (scoping params)
    params = urllib.parse.urlencode({"external_viewer_id": ext_viewer, "external_value": ext_value})
    token_info = _http(
        f"{INSTANCE_URL}/api/2.0/lakeview/dashboards/{DASHBOARD_ID}/published/tokeninfo?{params}",
        headers={"Authorization": f"Bearer {oidc_token}"},
    )

    # Step 3 — scoped token
    auth_details = token_info.pop("authorization_details", None)
    body_params  = {**token_info, "grant_type": "client_credentials"}
    if auth_details:
        body_params["authorization_details"] = json.dumps(auth_details)

    step3 = _http(
        f"{INSTANCE_URL}/oidc/v1/token",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {basic_auth}"},
        body=urllib.parse.urlencode(body_params),
    )
    return {"token": step3["access_token"]}


@router.get("/api/health")
async def health():
    return {
        "status":       "healthy",
        "models_loaded": get_loaded_models(),
        "db_ready":     bool(WAREHOUSE_ID),
        "warehouse_id": WAREHOUSE_ID,
    }
