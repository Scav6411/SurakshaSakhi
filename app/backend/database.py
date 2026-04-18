import os
import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState, Disposition
    from databricks.sdk.service.sql import Format as SqlFormat

    _client = WorkspaceClient()
    WAREHOUSE_ID: str = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    if not WAREHOUSE_ID:
        wh_list = list(_client.warehouses.list())
        WAREHOUSE_ID = wh_list[0].id if wh_list else ""
    logger.info(f"Databricks SDK ready — warehouse: {WAREHOUSE_ID}")
except Exception as exc:
    logger.warning(f"Databricks SDK unavailable: {exc}")
    _client = None
    WAREHOUSE_ID = ""
    StatementState = None
    Disposition = None
    SqlFormat = None


def run_query(sql: str) -> List[dict]:
    if not _client or not WAREHOUSE_ID:
        return []
    try:
        resp = _client.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            wait_timeout="30s",
            disposition=Disposition.INLINE,
            format=SqlFormat.JSON_ARRAY,
        )
        if resp.status.state != StatementState.SUCCEEDED:
            logger.error(f"SQL failed [{resp.status.state}]: {sql[:120]}")
            return []
        if not resp.result or not resp.manifest:
            return []
        cols = [c.name for c in resp.manifest.schema.columns]
        return [dict(zip(cols, row)) for row in (resp.result.data_array or [])]
    except Exception as exc:
        logger.error(f"SQL error: {exc}")
        return []


def execute(sql: str) -> None:
    if not _client or not WAREHOUSE_ID:
        return
    try:
        resp = _client.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            wait_timeout="50s",
            disposition=Disposition.INLINE,
            format=SqlFormat.JSON_ARRAY,
        )
        if resp.status.state != StatementState.SUCCEEDED:
            logger.error(f"DML failed [{resp.status.state}]: {resp.status.error} | SQL: {sql[:200]}")
            raise RuntimeError(f"DML failed: {resp.status.state} — {resp.status.error}")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.error(f"DML error: {exc}")
        raise


def esc(v) -> str:
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def init_tables() -> None:
    # workspace.suraksha.patients is created by notebook 09_create_app_table.py
    # and pre-populated with currently pregnant women from raw_survey.
    # We only ensure visit_log exists here.
    execute("""
        CREATE TABLE IF NOT EXISTS workspace.suraksha.visit_log (
            visit_id     STRING,
            patient_id   STRING,
            visit_date   STRING,
            notes        STRING,
            overall_risk DOUBLE
        ) USING DELTA
    """)
    logger.info("Tables ready")
