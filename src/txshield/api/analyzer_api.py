"""Optional API adapter. Install FastAPI and Uvicorn separately to enable it."""
def create_app(predictor):
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to enable the API") from exc
    app = FastAPI(title="TxShield API", version="0.1.0")

    @app.post("/predict")
    def predict(payload: dict):
        transaction = payload.get("transaction")
        if not isinstance(transaction, (str, dict)):
            return {"error": "'transaction' must be a JSON string or object"}
        import json
        text = transaction if isinstance(transaction, str) else json.dumps(transaction)
        return predictor.predict(text)
    return app
