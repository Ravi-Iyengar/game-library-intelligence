"""
Completion / status-signal features.

The project spec's status vocabulary is `completed / shelved / abandoned
/ playing / backlog / wishlist`. Your actual export uses a different,
overlapping-but-not-identical set: `completed / retired / played /
abandoned / shelved`. STATUS_SIGNAL_MAP below is the confirmed bridge
between them (confirmed directly with the library's owner):

    completed -> positive   (exact match with spec)
    abandoned -> negative   (exact match with spec)
    shelved   -> neutral    (exact match with spec)
    played    -> neutral    (an open-ended/no-defined-ending game —
                             e.g. Warframe, Overwatch, Minecraft — that
                             was played but isn't "finished" in the way
                             completed implies; not a preference signal
                             either way)
    retired   -> neutral    ("got my fill and am not planning to go
                             back" — distinct from abandoned, which is
                             a genuine negative signal (chose not to
                             continue))

If your own sense of these statuses ever changes, edit
STATUS_SIGNAL_MAP directly — everything downstream (feature store,
analytics, and eventually the Sprint 5 training labels) reads from it.
"""
from __future__ import annotations

STATUS_SIGNAL_MAP: dict[str, str] = {
    "completed": "positive",
    "abandoned": "negative",
    "shelved": "neutral",
    "played": "neutral",
    "retired": "neutral",
}


def signal_category(status: str | None) -> str:
    if status is None:
        return "unknown"
    return STATUS_SIGNAL_MAP.get(status, "unknown")


def compute_completion_features(
    status: str | None,
    is_backlog: int | None,
    is_playing: int | None,
    is_wishlist: int | None,
    is_liked: int | None,
    playthrough_count: int,
    replay_count: int,
) -> dict[str, int | float | str | None]:
    category = signal_category(status)
    return {
        "status_raw": status,
        "signal_category": category,
        "is_completed": int(status == "completed"),
        "is_abandoned": int(status == "abandoned"),
        "is_shelved": int(status == "shelved"),
        "is_backlog": int(bool(is_backlog)),
        "is_playing": int(bool(is_playing)),
        "is_wishlist": int(bool(is_wishlist)),
        "is_liked": int(bool(is_liked)),
        "replay_rate": (replay_count / playthrough_count) if playthrough_count else None,
    }
