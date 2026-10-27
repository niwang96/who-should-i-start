"""Stage 5: the web app. FastAPI renders Jinja templates; htmx (vendored, no CDN) swaps in results.

Every page works without JavaScript (forms post and get a full page back); with it, htmx
asks for just the result fragment. The app is a thin layer over the existing pieces:
`evidence.packet_for_names` builds the packet, `decide.decide` answers it within the
free-tier limits (one budgeted Gemini client per request, keyed to the visitor), and
`decide.record_recommendation` logs every answer with its packet for the evaluation.

Run locally: uvicorn startsit.web.app:app --reload  (uses $DATABASE_URL or the SQLite file)
"""

# No `from __future__ import annotations` here: FastAPI reads the endpoint annotations at
# runtime, including the `DB` alias defined inside create_app.
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from startsit.db import Database
from startsit.decide import HIGH_MARGIN, MEDIUM_MARGIN, Decision, decide, decide_llm, record_recommendation
from startsit.evaluate import CONFIDENCES, Report, build_report
from startsit.evidence import EvidencePacket, PlayerLookupError, current_week, eastern_time, packet_for_names
from startsit.llm import BudgetedLLM, visitor_id
from startsit.names import PlayerMatcher
from startsit.scoring import LABELS, SCORINGS, Scoring
from startsit.web import diag

log = logging.getLogger("startsit.web")

HERE = Path(__file__).parent
MAX_PLAYERS = 2  # the page compares two players (the engine takes more)
MAX_NAME_LENGTH = 60
PLAYER_CACHE_SECONDS = 3600  # the players table changes once a day, with the ingest
RECORD_CACHE_SECONDS = 3600  # results change once a day, when the stats refresh scores the new games

# The track record page's groups, in the order shown (keys from startsit.evaluate.group_of).
RECORD_GROUPS = [
    {
        "key": "benchmark",
        "title": "Weekly benchmark",
        "about": "30 close matchups a week (players the experts rank side by side), asked Sunday morning "
        "with that morning's news, like setting a lineup.",
    },
    {
        "key": "backfill",
        "title": "Earlier weeks (backfill)",
        "about": "The same kind of matchups, asked as of past Sunday mornings with the rankings from then. "
        "News collection started Oct 24, so most of these had no late news: a harder test for the AI.",
    },
    {"key": "users", "title": "Questions asked on this site", "about": "Every answer shown here, scored the same way."},
]

SOURCE_LABELS = {"llm": "AI pick", "cache": "AI pick", "baseline": "Baseline pick"}
KIND_LABELS = {
    "availability": "Availability",
    "consensus": "Expert consensus",
    "projection": "Projection",
    "matchup": "Matchup",
    "usage": "Usage",
    "injury": "Injury report",
    "news": "News",
}

# Same-origin only: htmx and the stylesheet are served from /static.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response


@dataclass
class WeekInfo:
    season: int
    week: int
    rankings_as_of: str | None
    news_as_of: str | None
    stats_as_of: str | None
    gamedays: tuple[str, str] | None = None  # the week's first and last game days


def game_day(day: str) -> str:
    """'2026-10-22' -> 'Thu Oct 22'."""
    d = date.fromisoformat(day)
    return f"{d:%a %b} {d.day}"


_CITATION = re.compile(r"\[?\b(E\d+)\b\]?")


def link_citations(text: str, known_ids: set[str]) -> Markup:
    """Escape `text` and turn evidence ids like [E2] into links to that item on the page."""

    def link(m: re.Match[str]) -> str:
        item = m.group(1)
        return f'<a class="cite" href="#{item}">{item}</a>' if item in known_ids else m.group(0)

    return Markup(_CITATION.sub(link, str(escape(text))))


def client_address(request: Request) -> str:
    """The visitor's IP: the first X-Forwarded-For hop behind a proxy (Vercel sets it), else the socket's."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def clean_names(players: list[str]) -> list[str]:
    return [p.strip()[:MAX_NAME_LENGTH] for p in players if p.strip()]


def create_app(
    db_target: str | None = None,
    llm_factory: Callable[[Database, str | None], BudgetedLLM | None] = decide_llm,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    diagnostics: bool | None = None,
) -> FastAPI:
    """The app. `diagnostics` adds /diag/limits; by default only on Vercel preview deployments."""
    if diagnostics is None:
        diagnostics = os.environ.get("VERCEL_ENV") == "preview"
    app = FastAPI(title="Who Should I Start?", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityHeaders)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(
        scorings=SCORINGS,
        scoring_labels=LABELS,
        source_labels=SOURCE_LABELS,
        kind_labels=KIND_LABELS,
        high_margin=HIGH_MARGIN,
        medium_margin=MEDIUM_MARGIN,
        confidences=CONFIDENCES,
    )
    templates.env.filters["eastern"] = eastern_time
    templates.env.filters["link_citations"] = link_citations
    templates.env.filters["game_day"] = game_day

    setup_done = False  # create missing tables once per process, not on every request
    matcher_cache: dict[str, Any] = {}
    record_cache: dict[int, tuple[float, Report]] = {}

    def get_db() -> Iterator[Database]:
        nonlocal setup_done
        db = Database(db_target, setup=not setup_done)
        setup_done = True
        try:
            yield db
        finally:
            db.close()

    DB = Annotated[Database, Depends(get_db)]

    def matcher(db: Database) -> PlayerMatcher:
        if time.monotonic() - matcher_cache.get("at", -PLAYER_CACHE_SECONDS) >= PLAYER_CACHE_SECONDS:
            matcher_cache.update(at=time.monotonic(), matcher=PlayerMatcher(db.all_players()))
        return cast(PlayerMatcher, matcher_cache["matcher"])

    def week_info(db: Database) -> WeekInfo | None:
        season = db.latest_season()
        if season is None:
            return None
        week = current_week(db, season)
        return WeekInfo(
            season,
            week,
            db.latest_scrape_date(season, week),
            db.latest_news_fetch(),
            db.latest_stats_run(),
            db.week_gamedays(season, week),
        )

    def page(request: Request, db: Database, **context: Any) -> HTMLResponse:
        context.setdefault("players", ["", ""])
        context.setdefault("scoring", "ppr")
        # The whole player list ships with the page (~1,000 names), so the browser suggests names
        # as you type without asking the server; nothing is requested until the button is pressed.
        return templates.TemplateResponse(
            request,
            "index.html",
            {"info": week_info(db), "max_players": MAX_PLAYERS, "rows": matcher(db).players, **context},
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, db: DB) -> HTMLResponse:
        return page(request, db)

    @app.post("/ask", response_class=HTMLResponse)
    def ask(
        request: Request,
        db: DB,
        players: Annotated[list[str], Form()] = [],  # noqa: B006 (FastAPI copies form defaults)
        scoring: Annotated[str, Form()] = "ppr",
    ) -> HTMLResponse:
        names = clean_names(players)[:MAX_PLAYERS]
        chosen: Scoring = scoring if scoring in SCORINGS else "ppr"
        context: dict[str, Any] = {"players": names + [""] * max(0, 2 - len(names)), "scoring": chosen}
        if len(names) < 2:
            context["error"] = "Enter at least two players."
        else:
            try:
                context |= answer(db, names, chosen, visitor_id(client_address(request)))
            except (PlayerLookupError, ValueError) as e:
                context["error"] = str(e)
            except Exception:  # anything else is our bug or a database hiccup: say so instead of a blank 500
                log.exception("answering %s failed", names)
                context["error"] = "Something went wrong on our side. Please try again in a minute."
        if request.headers.get("hx-request"):
            return templates.TemplateResponse(request, "_result.html", context)
        return page(request, db, **context)

    def answer(db: Database, names: list[str], scoring: Scoring, visitor: str) -> dict[str, Any]:
        info = week_info(db)
        if info is None:
            raise ValueError("No data yet. Please try again later.")
        packet: EvidencePacket = packet_for_names(
            db, names, scoring, info.season, info.week, matcher(db), as_of=clock()
        )
        if started := [p for p in packet.players if p.game_started]:
            # A player whose game has kicked off is locked in or out of the lineup in most leagues, so
            # there's no choice left: no AI call, and nothing logged for the evaluation.
            return {"packet": packet, "locked": started, "names": packet.names, "info": info}
        decision: Decision = decide(db, packet, llm_factory(db, visitor))
        try:
            record_recommendation(db, packet, decision, visitor)
        except Exception:  # logging must never cost the user their answer
            log.exception("recording the recommendation failed")
        return {"packet": packet, "decision": decision, "names": packet.names}

    @app.get("/record", response_class=HTMLResponse)
    def record(request: Request, db: DB) -> HTMLResponse:
        season = db.latest_season()
        report = None
        if season is not None:
            cached = record_cache.get(season)
            if cached is None or time.monotonic() - cached[0] >= RECORD_CACHE_SECONDS:
                cached = record_cache[season] = (time.monotonic(), build_report(db, season))
            report = cached[1]
        return templates.TemplateResponse(
            request, "record.html", {"info": week_info(db), "report": report, "groups": RECORD_GROUPS}
        )

    @app.get("/healthz")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "time": datetime.now(UTC).isoformat(timespec="seconds")})

    if diagnostics:

        @app.get("/diag/limits")
        def diag_limits(request: Request, db: DB) -> JSONResponse:
            return JSONResponse(diag.limits_report(db, request.headers, client_address(request), clock()))

    return app


app = create_app()
