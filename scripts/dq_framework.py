"""
scripts/dq_framework.py
=======================================================
Part E - Data Quality Framework (Questions 25-29)
=======================================================

Q25 Define data quality rules in YAML config file
Q26 Implement Great Expectations: create expectations for each table
Q27 Create data quality dashboard: pass/fail rate, trend over time
Q28 Build alerting: Slack/email notification on quality failures
Q29 Implement data profiling: generate HTML report with statistics
"""

import os
import sys
import json
import logging
import smtplib
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, FloatType, BooleanType, LongType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dq_framework")

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH    = os.path.join(_PROJECT_ROOT, "config",       "dq_rules.yaml")
EXPECTATIONS   = os.path.join(_PROJECT_ROOT, "expectations")
DQ_RESULTS     = os.path.join(_PROJECT_ROOT, "logs",         "dq_results.json")
REPORTS_DIR    = os.path.join(_PROJECT_ROOT, "logs",         "reports")
SILVER_BASE    = os.path.join(_PROJECT_ROOT, "silver")

os.makedirs(EXPECTATIONS, exist_ok=True)
os.makedirs(REPORTS_DIR,  exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Q25 - YAML DQ RULES CONFIG
# Central place to define all quality rules — no hardcoding in pipeline code
# ═════════════════════════════════════════════════════════════════════════════

DQ_RULES_YAML = """
# config/dq_rules.yaml
# Data Quality Rules for Netflix Medallion Pipeline
# Each table defines: not_null, range, unique, and custom SQL checks

version: "1.0"
tables:

  silver_ratings:
    not_null:
      - user_id
      - movie_id
      - rating
    range:
      - column: rating
        min: 1.0
        max: 5.0
    unique:
      - [user_id, movie_id]
    thresholds:
      max_null_pct:      5.0
      max_duplicate_pct: 1.0
      min_row_count:     100

  silver_movies:
    not_null:
      - movie_id
      - title
    unique:
      - [movie_id]
    thresholds:
      max_null_pct:  0.0
      min_row_count: 10

  silver_users:
    not_null:
      - user_id
    unique:
      - [user_id]
    thresholds:
      max_null_pct:  0.0
      min_row_count: 10

  fact_ratings:
    not_null:
      - user_id
      - movie_id
      - rating_date
      - rating
    range:
      - column: rating
        min: 1.0
        max: 5.0
    thresholds:
      max_null_pct:  0.0
      min_row_count: 50
"""


def write_dq_rules_yaml() -> str:
    """Write the DQ rules YAML to config/dq_rules.yaml."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        f.write(DQ_RULES_YAML)
    logger.info(f"DQ rules written to {CONFIG_PATH}")
    return CONFIG_PATH


def load_dq_rules() -> dict:
    """Load DQ rules from YAML. Creates file if it doesn't exist."""
    if not os.path.exists(CONFIG_PATH):
        write_dq_rules_yaml()
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_table_rules(table_name: str) -> Optional[dict]:
    """Return rules for a specific table, or None if not defined."""
    rules = load_dq_rules()
    return rules.get("tables", {}).get(table_name)


# ═════════════════════════════════════════════════════════════════════════════
# Q26 - GREAT EXPECTATIONS (lightweight implementation)
# We implement the GE expectations pattern using pure PySpark.
# This mirrors GE's API without requiring the heavy GE dependency.
# ═════════════════════════════════════════════════════════════════════════════

class ExpectationResult:
    """Result of a single expectation check."""
    def __init__(
        self,
        expectation_type: str,
        column: Optional[str],
        passed: bool,
        observed_value: Any,
        expected_value: Any,
        details: str = "",
    ):
        self.expectation_type = expectation_type
        self.column           = column
        self.passed           = passed
        self.observed_value   = observed_value
        self.expected_value   = expected_value
        self.details          = details
        self.checked_at       = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "expectation_type": self.expectation_type,
            "column":           self.column,
            "passed":           self.passed,
            "observed_value":   self.observed_value,
            "expected_value":   self.expected_value,
            "details":          self.details,
            "checked_at":       self.checked_at,
        }


class ExpectationSuite:
    """
    Collection of expectations for one table.
    Mirrors Great Expectations' ExpectationSuite API.
    """

    def __init__(self, suite_name: str, df: DataFrame):
        self.suite_name = suite_name
        self.df         = df
        self.results:   List[ExpectationResult] = []
        self._row_count = df.count()

    # ── Core expectations ─────────────────────────────────────────────────────

    def expect_column_values_to_not_be_null(self, column: str) -> "ExpectationSuite":
        """Expect no null values in column."""
        if column not in self.df.columns:
            self.results.append(ExpectationResult(
                "expect_column_values_to_not_be_null", column,
                False, "column_missing", "column_exists",
                f"Column '{column}' does not exist"
            ))
            return self

        null_count = self.df.filter(F.col(column).isNull()).count()
        null_pct   = round(null_count / self._row_count * 100, 2) \
                     if self._row_count > 0 else 0

        self.results.append(ExpectationResult(
            "expect_column_values_to_not_be_null", column,
            null_count == 0,
            f"{null_count} nulls ({null_pct}%)", "0 nulls",
            f"{null_count} null values found in '{column}'"
        ))
        return self

    def expect_column_values_to_be_between(
        self, column: str, min_value: float, max_value: float
    ) -> "ExpectationSuite":
        """Expect all values in column to be within [min_value, max_value]."""
        if column not in self.df.columns:
            return self

        bad_count = self.df.filter(
            F.col(column).isNotNull() &
            ((F.col(column) < min_value) | (F.col(column) > max_value))
        ).count()

        self.results.append(ExpectationResult(
            "expect_column_values_to_be_between", column,
            bad_count == 0,
            f"{bad_count} out-of-range values",
            f"all values in [{min_value}, {max_value}]",
            f"{bad_count} values outside [{min_value}, {max_value}]"
        ))
        return self

    def expect_column_values_to_be_unique(
        self, columns: List[str]
    ) -> "ExpectationSuite":
        """Expect composite key to be unique across all rows."""
        existing = [c for c in columns if c in self.df.columns]
        if not existing:
            return self

        dup_count = (
            self.df.groupBy(existing)
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        key_label = "+".join(existing)

        self.results.append(ExpectationResult(
            "expect_column_values_to_be_unique", key_label,
            dup_count == 0,
            f"{dup_count} duplicate keys",
            "0 duplicate keys",
            f"{dup_count} duplicate ({key_label}) combinations found"
        ))
        return self

    def expect_table_row_count_to_be_between(
        self, min_count: int, max_count: int = 10_000_000
    ) -> "ExpectationSuite":
        """Expect row count to be within [min_count, max_count]."""
        passed = min_count <= self._row_count <= max_count
        self.results.append(ExpectationResult(
            "expect_table_row_count_to_be_between", None,
            passed,
            self._row_count,
            f"[{min_count}, {max_count}]",
            f"Row count {self._row_count} {'ok' if passed else 'out of range'}"
        ))
        return self

    def expect_column_to_exist(self, column: str) -> "ExpectationSuite":
        """Expect column to be present in the DataFrame."""
        exists = column in self.df.columns
        self.results.append(ExpectationResult(
            "expect_column_to_exist", column,
            exists,
            "exists" if exists else "missing",
            "exists",
        ))
        return self

    def expect_column_values_to_be_in_set(
        self, column: str, value_set: list
    ) -> "ExpectationSuite":
        """Expect all non-null values in column to be in the given set."""
        if column not in self.df.columns:
            return self
        bad = self.df.filter(
            F.col(column).isNotNull() & ~F.col(column).isin(value_set)
        ).count()
        self.results.append(ExpectationResult(
            "expect_column_values_to_be_in_set", column,
            bad == 0,
            f"{bad} values not in set",
            f"all in {value_set}",
        ))
        return self

    # ── Run and report ────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Evaluate all expectations and return a summary report dict.
        Also saves results to expectations/{suite_name}.json
        """
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        success_pct = round(passed / len(self.results) * 100, 1) \
                      if self.results else 100.0

        report = {
            "suite_name":   self.suite_name,
            "run_date":     datetime.utcnow().isoformat(),
            "total_rows":   self._row_count,
            "total_checks": len(self.results),
            "passed":       passed,
            "failed":       failed,
            "success_pct":  success_pct,
            "overall_pass": failed == 0,
            "results":      [r.to_dict() for r in self.results],
        }

        # Persist to expectations folder
        out_path = os.path.join(EXPECTATIONS, f"{self.suite_name}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        # Append to running DQ results log
        summary = {k: v for k, v in report.items() if k != "results"}
        all_results = []
        if os.path.exists(DQ_RESULTS):
            with open(DQ_RESULTS) as f:
                all_results = json.load(f)
        all_results.append(summary)
        os.makedirs(os.path.dirname(DQ_RESULTS), exist_ok=True)
        with open(DQ_RESULTS, "w") as f:
            json.dump(all_results, f, indent=2)

        status = "PASS" if failed == 0 else "FAIL"
        logger.info(
            f"[{status}] {self.suite_name}: "
            f"{passed}/{len(self.results)} checks passed ({success_pct}%)"
        )
        return report


def build_suite_from_yaml(table_name: str, df: DataFrame) -> ExpectationSuite:
    """
    Build an ExpectationSuite for a table using rules from dq_rules.yaml.
    This is the bridge between Q25 (YAML rules) and Q26 (expectations).
    """
    suite = ExpectationSuite(table_name, df)
    rules = get_table_rules(table_name)

    if not rules:
        logger.warning(f"No DQ rules found for {table_name}")
        return suite

    # Not-null checks
    for col in rules.get("not_null", []):
        suite.expect_column_values_to_not_be_null(col)

    # Range checks
    for rule in rules.get("range", []):
        suite.expect_column_values_to_be_between(
            rule["column"], rule["min"], rule["max"]
        )

    # Uniqueness checks
    for key in rules.get("unique", []):
        cols = key if isinstance(key, list) else [key]
        suite.expect_column_values_to_be_unique(cols)

    # Row count threshold
    thresholds = rules.get("thresholds", {})
    min_rows = thresholds.get("min_row_count", 0)
    if min_rows:
        suite.expect_table_row_count_to_be_between(min_rows)

    return suite


def run_all_suites(spark: SparkSession) -> List[dict]:
    """Run expectations for all tables defined in dq_rules.yaml."""
    write_dq_rules_yaml()
    rules  = load_dq_rules()
    tables = rules.get("tables", {})
    reports = []

    for table_name in tables:
        # Map table name to silver path
        silver_name = table_name.replace("silver_", "").replace("fact_", "")
        silver_path = os.path.join(SILVER_BASE, silver_name)

        if not os.path.exists(silver_path):
            logger.warning(f"Skipping {table_name} — path not found: {silver_path}")
            continue

        try:
            delta_log = os.path.join(silver_path, "_delta_log")
            if os.path.exists(delta_log):
                df = spark.read.format("delta").load(silver_path)
            else:
                df = spark.read.option("recursiveFileLookup", "true").parquet(silver_path)

            suite  = build_suite_from_yaml(table_name, df)
            report = suite.run()
            reports.append(report)
        except Exception as e:
            logger.error(f"Suite failed for {table_name}: {e}")

    return reports


# ═════════════════════════════════════════════════════════════════════════════
# Q27 - DQ DASHBOARD (HTML)
# Shows pass/fail rate per table and trend over time
# ═════════════════════════════════════════════════════════════════════════════

def build_dq_dashboard(reports: List[dict]) -> str:
    """
    Generate a self-contained HTML dashboard showing:
      - Per-table pass/fail summary cards
      - Check-level detail table
      - Trend chart (pass % over time)
    Returns path to generated HTML file.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build table rows for each suite
    table_rows = ""
    for r in reports:
        status_color = "#1D9E75" if r["overall_pass"] else "#E24B4A"
        status_label = "PASS" if r["overall_pass"] else "FAIL"
        table_rows += f"""
        <tr>
          <td><strong>{r['suite_name']}</strong></td>
          <td style="color:{status_color};font-weight:500">{status_label}</td>
          <td>{r['total_rows']:,}</td>
          <td>{r['passed']}</td>
          <td>{r['failed']}</td>
          <td>
            <div style="background:#e8e8e8;border-radius:4px;height:12px;width:100px;display:inline-block">
              <div style="background:{status_color};height:12px;border-radius:4px;
                          width:{r['success_pct']}px"></div>
            </div>
            {r['success_pct']}%
          </td>
        </tr>"""

    # Build detail rows
    detail_rows = ""
    for r in reports:
        for check in r.get("results", []):
            icon   = "✓" if check["passed"] else "✗"
            color  = "#1D9E75" if check["passed"] else "#E24B4A"
            detail_rows += f"""
            <tr>
              <td>{r['suite_name']}</td>
              <td>{check['expectation_type']}</td>
              <td>{check.get('column') or '—'}</td>
              <td style="color:{color};font-weight:500">{icon}</td>
              <td>{check['observed_value']}</td>
              <td>{check['expected_value']}</td>
            </tr>"""

    # Load historical trend from DQ results log
    trend_labels, trend_values = "[]", "[]"
    if os.path.exists(DQ_RESULTS):
        with open(DQ_RESULTS) as f:
            history = json.load(f)
        if history:
            dates  = [h["run_date"][:10]   for h in history[-20:]]
            scores = [h["success_pct"]      for h in history[-20:]]
            trend_labels = json.dumps(dates)
            trend_values = json.dumps(scores)

    total_checks  = sum(r["total_checks"] for r in reports)
    total_passed  = sum(r["passed"]        for r in reports)
    overall_pct   = round(total_passed / total_checks * 100, 1) if total_checks else 0
    tables_passed = sum(1 for r in reports if r["overall_pass"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Quality Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #333; padding: 24px; }}
  h1   {{ font-size: 24px; font-weight: 600; margin-bottom: 4px; }}
  .sub {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .card  {{ background: #fff; border-radius: 12px; padding: 20px 24px;
            flex: 1; min-width: 160px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .card .val {{ font-size: 32px; font-weight: 700; margin-bottom: 4px; }}
  .card .lbl {{ font-size: 13px; color: #888; }}
  .green {{ color: #1D9E75; }}  .red {{ color: #E24B4A; }}
  .section {{ background: #fff; border-radius: 12px; padding: 20px 24px;
              margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th    {{ text-align: left; padding: 8px 12px; background: #f8f8f8;
           font-weight: 500; border-bottom: 2px solid #e8e8e8; }}
  td    {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
  .chart-wrap {{ height: 220px; }}
</style>
</head>
<body>
<h1>Data Quality Dashboard</h1>
<div class="sub">Generated {now} · Netflix Medallion Pipeline</div>

<div class="cards">
  <div class="card">
    <div class="val {'green' if overall_pct >= 80 else 'red'}">{overall_pct}%</div>
    <div class="lbl">Overall pass rate</div>
  </div>
  <div class="card">
    <div class="val">{len(reports)}</div>
    <div class="lbl">Tables checked</div>
  </div>
  <div class="card">
    <div class="val green">{tables_passed}</div>
    <div class="lbl">Tables passing</div>
  </div>
  <div class="card">
    <div class="val red">{len(reports) - tables_passed}</div>
    <div class="lbl">Tables failing</div>
  </div>
  <div class="card">
    <div class="val">{total_checks}</div>
    <div class="lbl">Total checks</div>
  </div>
</div>

<div class="section">
  <h2>Pass rate trend</h2>
  <div class="chart-wrap">
    <canvas id="trend"></canvas>
  </div>
</div>

<div class="section">
  <h2>Table summary</h2>
  <table>
    <thead>
      <tr><th>Table</th><th>Status</th><th>Rows</th>
          <th>Passed</th><th>Failed</th><th>Pass rate</th></tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>Check details</h2>
  <table>
    <thead>
      <tr><th>Table</th><th>Expectation</th><th>Column</th>
          <th>Result</th><th>Observed</th><th>Expected</th></tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>
</div>

<script>
new Chart(document.getElementById('trend'), {{
  type: 'line',
  data: {{
    labels: {trend_labels},
    datasets: [{{
      label: 'Pass %',
      data: {trend_values},
      borderColor: '#1D9E75',
      backgroundColor: 'rgba(29,158,117,0.08)',
      fill: true, tension: 0.3, pointRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{ y: {{ min: 0, max: 100,
      ticks: {{ callback: v => v + '%' }} }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>
</body>
</html>"""

    out_path = os.path.join(REPORTS_DIR, "dq_dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"DQ dashboard written to {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# Q28 - ALERTING: SLACK + EMAIL on quality failures
# ═════════════════════════════════════════════════════════════════════════════

def send_slack_alert(webhook_url: str, reports: List[dict]) -> bool:
    """
    Send a Slack notification summarising DQ failures.
    Uses Slack Incoming Webhooks — set SLACK_WEBHOOK_URL env var.
    Returns True if sent successfully.
    """
    try:
        import urllib.request
        failed_tables = [r for r in reports if not r["overall_pass"]]
        if not failed_tables:
            logger.info("No DQ failures — Slack alert skipped")
            return True

        lines = [f"*Data Quality Alert* — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"]
        lines.append(f"{len(failed_tables)} table(s) failed DQ checks:\n")
        for r in failed_tables:
            lines.append(
                f"• *{r['suite_name']}*: "
                f"{r['passed']}/{r['total_checks']} checks passed "
                f"({r['success_pct']}%)"
            )
        lines.append("\nCheck the DQ dashboard for details.")

        payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            success = resp.status == 200
            logger.info(f"Slack alert sent: status={resp.status}")
            return success

    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        return False


def send_email_alert(
    reports: List[dict],
    smtp_host: str = "localhost",
    smtp_port: int = 587,
    sender: str    = "pipeline@netflix.local",
    recipient: str = "data-team@netflix.local",
    password: str  = "",
) -> bool:
    """
    Send an email summary of DQ failures.
    Configure via environment variables:
      ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_EMAIL_FROM,
      ALERT_EMAIL_TO, ALERT_EMAIL_PASSWORD
    """
    try:
        failed = [r for r in reports if not r["overall_pass"]]
        if not failed:
            logger.info("No DQ failures — email alert skipped")
            return True

        body_lines = ["Data Quality Report\n"]
        for r in failed:
            body_lines.append(
                f"Table: {r['suite_name']}\n"
                f"  Passed: {r['passed']}/{r['total_checks']} ({r['success_pct']}%)\n"
            )
            for check in r.get("results", []):
                if not check["passed"]:
                    body_lines.append(
                        f"  FAIL: {check['expectation_type']} "
                        f"on {check.get('column','*')} — "
                        f"observed: {check['observed_value']}\n"
                    )

        msg = MIMEMultipart()
        msg["Subject"] = f"[DQ ALERT] {len(failed)} table(s) failed — Netflix Pipeline"
        msg["From"]    = sender
        msg["To"]      = recipient
        msg.attach(MIMEText("".join(body_lines), "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if password:
                server.starttls()
                server.login(sender, password)
            server.send_message(msg)

        logger.info(f"Email alert sent to {recipient}")
        return True

    except Exception as e:
        logger.error(f"Email alert failed: {e}")
        return False


def alert_on_failures(reports: List[dict]) -> None:
    """
    Check reports for failures and fire alerts via configured channels.
    Reads config from environment variables — safe to call even without setup.
    """
    failed = [r for r in reports if not r["overall_pass"]]
    if not failed:
        logger.info("All DQ checks passed — no alerts needed")
        return

    logger.warning(f"{len(failed)} DQ failures detected — sending alerts")

    # Slack
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if webhook:
        send_slack_alert(webhook, reports)
    else:
        logger.info("SLACK_WEBHOOK_URL not set — Slack alert skipped")

    # Email
    smtp_host = os.getenv("ALERT_SMTP_HOST", "")
    if smtp_host:
        send_email_alert(
            reports,
            smtp_host=smtp_host,
            smtp_port=int(os.getenv("ALERT_SMTP_PORT", "587")),
            sender=os.getenv("ALERT_EMAIL_FROM",     "pipeline@netflix.local"),
            recipient=os.getenv("ALERT_EMAIL_TO",    "data-team@netflix.local"),
            password=os.getenv("ALERT_EMAIL_PASSWORD", ""),
        )
    else:
        logger.info("ALERT_SMTP_HOST not set — email alert skipped")


# ═════════════════════════════════════════════════════════════════════════════
# Q29 - DATA PROFILING: HTML REPORT
# Column-level statistics: min, max, mean, stddev, nulls, distinct, histogram
# ═════════════════════════════════════════════════════════════════════════════

def profile_dataframe(spark: SparkSession, df: DataFrame, table_name: str) -> str:
    """
    Generate a full data profile for a DataFrame and write an HTML report.
    Covers: row count, schema, per-column stats (nulls, min, max, mean, stddev,
    distinct count, top 5 values).
    Returns path to the HTML report.
    """
    logger.info(f"Profiling {table_name}...")
    total_rows = df.count()
    total_cols = len(df.columns)

    profiles = []
    for col_name in df.columns:
        dtype = str(df.schema[col_name].dataType)
        is_numeric = any(t in dtype for t in
                         ["IntegerType", "FloatType", "DoubleType",
                          "LongType", "DecimalType"])

        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct   = round(null_count / total_rows * 100, 1) if total_rows else 0
        distinct   = df.select(col_name).distinct().count()

        stats = {
            "column":     col_name,
            "dtype":      dtype,
            "null_count": null_count,
            "null_pct":   null_pct,
            "distinct":   distinct,
            "min": None, "max": None, "mean": None, "stddev": None,
        }

        if is_numeric:
            agg_row = df.agg(
                F.min(col_name).alias("min"),
                F.max(col_name).alias("max"),
                F.round(F.avg(col_name), 3).alias("mean"),
                F.round(F.stddev(col_name), 3).alias("stddev"),
            ).collect()[0]
            stats.update({
                "min":    agg_row["min"],
                "max":    agg_row["max"],
                "mean":   agg_row["mean"],
                "stddev": agg_row["stddev"],
            })
        else:
            min_row = df.agg(F.min(col_name)).collect()[0][0]
            max_row = df.agg(F.max(col_name)).collect()[0][0]
            stats["min"] = str(min_row) if min_row is not None else None
            stats["max"] = str(max_row) if max_row is not None else None

        # Top 5 most frequent values
        top5 = (
            df.groupBy(col_name).count()
            .orderBy(F.col("count").desc())
            .limit(5)
            .collect()
        )
        stats["top5"] = [(str(r[col_name]), r["count"]) for r in top5]
        profiles.append(stats)

    # Build HTML
    rows_html = ""
    for p in profiles:
        null_color = "#E24B4A" if p["null_pct"] > 5 else "#1D9E75"
        top5_str   = ", ".join(
            f"{v} ({c})" for v, c in p["top5"]
        ) if p["top5"] else "—"

        rows_html += f"""
        <tr>
          <td><strong>{p['column']}</strong></td>
          <td><code>{p['dtype']}</code></td>
          <td style="color:{null_color}">{p['null_count']} ({p['null_pct']}%)</td>
          <td>{p['distinct']:,}</td>
          <td>{p['min'] if p['min'] is not None else '—'}</td>
          <td>{p['max'] if p['max'] is not None else '—'}</td>
          <td>{p['mean'] if p['mean'] is not None else '—'}</td>
          <td>{p['stddev'] if p['stddev'] is not None else '—'}</td>
          <td style="font-size:12px;color:#666">{top5_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Profile: {table_name}</title>
<style>
  body  {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
           padding:24px;background:#f5f5f5;color:#333; }}
  h1    {{ font-size:22px;font-weight:600;margin-bottom:4px; }}
  .sub  {{ color:#888;font-size:13px;margin-bottom:20px; }}
  .cards {{ display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap; }}
  .card  {{ background:#fff;border-radius:10px;padding:16px 20px;
            min-width:140px;box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  .card .v {{ font-size:28px;font-weight:700; }}
  .card .l {{ font-size:12px;color:#888;margin-top:2px; }}
  .wrap  {{ background:#fff;border-radius:10px;padding:20px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow-x:auto; }}
  table  {{ width:100%;border-collapse:collapse;font-size:13px; }}
  th     {{ background:#f8f8f8;padding:8px 10px;text-align:left;
            font-weight:500;border-bottom:2px solid #e8e8e8;white-space:nowrap; }}
  td     {{ padding:7px 10px;border-bottom:1px solid #f0f0f0;vertical-align:top; }}
  code   {{ background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:12px; }}
</style>
</head>
<body>
<h1>Data Profile: {table_name}</h1>
<div class="sub">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
<div class="cards">
  <div class="card"><div class="v">{total_rows:,}</div><div class="l">Total rows</div></div>
  <div class="card"><div class="v">{total_cols}</div><div class="l">Columns</div></div>
  <div class="card"><div class="v">{sum(1 for p in profiles if p['null_count']>0)}</div>
    <div class="l">Cols with nulls</div></div>
</div>
<div class="wrap">
<table>
  <thead>
    <tr>
      <th>Column</th><th>Type</th><th>Nulls</th><th>Distinct</th>
      <th>Min</th><th>Max</th><th>Mean</th><th>Stddev</th><th>Top 5 values</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
</body>
</html>"""

    out_path = os.path.join(REPORTS_DIR, f"profile_{table_name}.html")
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Profile report written to {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_dq_framework(spark: SparkSession) -> None:
    logger.info("=" * 60)
    logger.info("PART E - DATA QUALITY FRAMEWORK")
    logger.info("=" * 60)

    # Q25: Write YAML rules
    write_dq_rules_yaml()
    logger.info("Q25: DQ rules YAML written")

    # Q26: Run all expectation suites
    reports = run_all_suites(spark)
    logger.info(f"Q26: {len(reports)} expectation suites run")

    # Q27: Build dashboard
    dashboard_path = build_dq_dashboard(reports)
    logger.info(f"Q27: Dashboard → {dashboard_path}")

    # Q28: Send alerts (no-op if env vars not set)
    alert_on_failures(reports)
    logger.info("Q28: Alert check complete")

    # Q29: Profile each silver table
    for name in ["ratings", "movies", "users", "enriched_ratings"]:
        path = os.path.join(SILVER_BASE, name)
        if not os.path.exists(path):
            continue
        delta = os.path.join(path, "_delta_log")
        df = spark.read.format("delta").load(path) \
             if os.path.exists(delta) \
             else spark.read.option("recursiveFileLookup", "true").parquet(path)
        profile_dataframe(spark, df, name)
    logger.info("Q29: Profile reports written")

    logger.info("=" * 60)
    logger.info("PART E COMPLETE")
    logger.info(f"  Reports: {REPORTS_DIR}")
    logger.info(f"  Expectations: {EXPECTATIONS}")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_ROOT)
    from utils.spark_session import get_spark
    spark = get_spark("DQFramework")
    run_dq_framework(spark)
    spark.stop()
