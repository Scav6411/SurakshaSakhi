import os
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient

router = APIRouter(prefix="/api/genie", tags=["genie"])

SPACE_ID = "01f13b07fc7515a6baf7db7d21088387"


class GenieQuery(BaseModel):
    query: str
    conversation_id: str | None = None


def _client() -> WorkspaceClient:
    # Databricks Apps injects DATABRICKS_HOST + service principal creds automatically.
    # Locally, fall back to ~/.databrickscfg profile Ayush.
    if os.environ.get("DATABRICKS_HOST"):
        return WorkspaceClient()
    return WorkspaceClient(profile="Ayush")


def _poll(w: WorkspaceClient, conv_id: str, msg_id: str, timeout: int = 40) -> dict:
    path     = f"/api/2.0/genie/spaces/{SPACE_ID}/conversations/{conv_id}/messages/{msg_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = w.api_client.do("GET", path)
        if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            return data
        time.sleep(2)
    raise TimeoutError("Genie timed out")


def _fetch_rows(w: WorkspaceClient, statement_id: str) -> dict | None:
    data = w.api_client.do("GET", f"/api/2.0/sql/statements/{statement_id}")
    if data.get("status", {}).get("state") != "SUCCEEDED":
        return None
    columns = [c["name"] for c in data["manifest"]["schema"].get("columns", [])]
    rows    = data["result"].get("data_array", [])
    return {"columns": columns, "rows": rows}


@router.post("/query")
def genie_query(body: GenieQuery):
    try:
        w = _client()

        if body.conversation_id:
            path = f"/api/2.0/genie/spaces/{SPACE_ID}/conversations/{body.conversation_id}/messages"
        else:
            path = f"/api/2.0/genie/spaces/{SPACE_ID}/start-conversation"

        data    = w.api_client.do("POST", path, body={"content": body.query})
        conv_id = data["conversation_id"]
        msg_id  = data["message_id"]

        msg = _poll(w, conv_id, msg_id)
        if msg.get("status") != "COMPLETED":
            raise HTTPException(status_code=500, detail="Genie query failed or was cancelled")

        text_parts   = []
        statement_id = None
        sql          = None

        for att in msg.get("attachments", []):
            if "text" in att:
                text_parts.append(att["text"]["content"])
            if "query" in att:
                statement_id = att["query"].get("statement_id")
                sql          = att["query"].get("query")

        text       = " ".join(text_parts) or "No answer found."
        table_data = _fetch_rows(w, statement_id) if statement_id else None

        return {
            "text":            text,
            "columns":         table_data["columns"] if table_data else None,
            "rows":            table_data["rows"]    if table_data else None,
            "sql":             sql,
            "conversation_id": conv_id,
        }

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Genie query timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
