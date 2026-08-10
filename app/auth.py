from fastapi import Request, HTTPException  # Request exposes headers and app.state; HTTPException returns a clean 401


async def require_api_key(request: Request) -> None:  # PURPOSE: gate a route behind the API key; returns None if the caller is allowed, raises 401 if not
    config = request.app.state.config  # single Config instance stored at startup, no re-fetch per request
    if not config.api_key:
        return  # auth disabled when no key is configured
    key = request.headers.get("X-API-Key", "")  # default "" so a missing header fails the same way as a wrong one
    if key != config.api_key:  # exact match required
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")  # same message for both cases, no hint to attackers


# Purpose: API key authentication for the service, written as a FastAPI dependency so any
# route can require it with Depends(require_api_key). It reads the expected key from the
# Config kept on app.state and compares it to the X-API-Key request header, raising 401 on
# any mismatch. Missing and wrong keys return the identical message, so an attacker learns
# nothing about which part failed. If API_KEY is empty in the secret, auth is skipped
# entirely, which is convenient for local development but must never be the case in production.
