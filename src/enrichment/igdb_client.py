"""
Thin IGDB v4 API client.

Handles exactly two things, on purpose:
  1. Twitch OAuth (client_credentials grant) -> Bearer token, cached
     until it expires.
  2. Rate-limited POST requests to api.igdb.com/v4/<endpoint> using the
     Apicalypse query language.

Credentials are read from environment variables (IGDB_CLIENT_ID /
IGDB_CLIENT_SECRET) — never hardcode them in source or pass them as
CLI args (they'd end up in shell history).

This module makes real network calls and is not something the ingestion
tests exercise; see tests/test_enrichment.py for how the parsing logic
downstream of this client is tested with a mocked client instead.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_BASE_URL = "https://api.igdb.com/v4"

# IGDB's documented limit is 4 requests/second. We space calls out at a
# slightly more conservative interval so small jitter doesn't tip us
# over into a 429.
MIN_SECONDS_BETWEEN_REQUESTS = 0.3
MAX_RETRIES = 5


class IGDBAuthError(RuntimeError):
    pass


@dataclass
class IGDBClient:
    client_id: str
    client_secret: str
    _access_token: str | None = field(default=None, init=False, repr=False)
    _token_expires_at: float = field(default=0.0, init=False, repr=False)
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_env(cls) -> "IGDBClient":
        client_id = os.environ.get("IGDB_CLIENT_ID")
        client_secret = os.environ.get("IGDB_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise IGDBAuthError(
                "Set IGDB_CLIENT_ID and IGDB_CLIENT_SECRET as environment "
                "variables before running enrichment (see docs/ROADMAP.md)."
            )
        return cls(client_id=client_id, client_secret=client_secret)

    def _ensure_token(self) -> None:
        if self._access_token and time.monotonic() < self._token_expires_at:
            return

        resp = requests.post(
            TWITCH_TOKEN_URL,
            params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise IGDBAuthError(
                f"Twitch token request failed ({resp.status_code}): {resp.text}"
            )

        payload = resp.json()
        self._access_token = payload["access_token"]
        # Refresh a little early rather than exactly at expiry.
        self._token_expires_at = time.monotonic() + payload.get("expires_in", 3600) - 60
        logger.info("Obtained IGDB access token (expires in ~%ss).", payload.get("expires_in"))

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

    def query(self, endpoint: str, apicalypse_body: str) -> list[dict]:
        """
        POST an Apicalypse query to an IGDB endpoint, e.g.:

            client.query("games", "fields name, genres.name; where id = (1,2,3); limit 500;")

        Retries on 429 with exponential backoff. Returns the parsed JSON
        list of results.
        """
        self._ensure_token()
        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            self._last_request_at = time.monotonic()

            resp = requests.post(
                f"{IGDB_BASE_URL}/{endpoint}",
                headers=headers,
                data=apicalypse_body,
                timeout=30,
            )

            if resp.status_code == 429:
                wait = 2**attempt
                logger.warning("IGDB rate limit hit, retrying in %ss (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                # Token may have been revoked/expired early — force a refresh once.
                self._access_token = None
                self._ensure_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                continue

            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"IGDB query to '{endpoint}' failed after {MAX_RETRIES} retries.")
