"""
Interactive Web Application Server for CreditIQ Loan Underwriting System.
Uses Python's standard library http.server with JSON REST endpoints.
Zero external server dependencies required.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import numpy as np

from src.config import BASE_DIR, FIGURES_DIR
from src.predict import LoanRiskPredictor
from src.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", 8501))
predictor = LoanRiskPredictor()


class UnderwritingRequestHandler(SimpleHTTPRequestHandler):
    """
    Request handler serving dashboard UI, image assets, and JSON prediction APIs.
    """

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            template_path = BASE_DIR / "web_app" / "templates" / "index.html"
            with open(template_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        elif path.startswith("/reports/figures/"):
            filename = Path(path).name
            img_path = FIGURES_DIR / filename
            if img_path.exists():
                with open(img_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            else:
                self.send_error(404, "Figure not found")
                return

        elif path == "/api/decisions":
            decisions_df = db_manager.get_recent_decisions(limit=25)
            records = decisions_df.to_dict(orient="records")
            # Convert non-serializable objects
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
                    elif isinstance(v, (np.floating, float)):
                        r[k] = round(float(v), 4)
            data = json.dumps(records).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        elif path == "/api/kpis":
            kpis = db_manager.get_portfolio_kpis()
            data = json.dumps(kpis).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                applicant_data = json.loads(body)
                notes = applicant_data.pop("notes", "Scored via Underwriting Web Portal")
                sk_id = applicant_data.get("SK_ID_CURR", int(np.random.randint(100000, 999999)))
                applicant_data["SK_ID_CURR"] = sk_id

                result = predictor.predict_single(applicant_data)

                # Persist decision to database
                db_manager.log_underwriting_decision(
                    sk_id_curr=sk_id,
                    default_probability=result["default_probability"],
                    risk_tier=result["risk_tier"],
                    recommendation=result["recommendation"],
                    decision_threshold=result["decision_cutoff"],
                    key_drivers=result.get("key_risk_factors", []),
                    notes=notes
                )

                response_data = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
            except Exception as e:
                logger.error(f"Prediction failed: {e}", exc_info=True)
                err_data = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)


def run_server(port: int = PORT):
    server = HTTPServer(("0.0.0.0", port), UnderwritingRequestHandler)
    print(f"\n=======================================================")
    print(f" CreditIQ Underwriting Web Portal running on:")
    print(f" http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
