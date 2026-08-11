import asyncio  # to_thread, keeps the blocking embedding call off the event loop
import difflib  # unified_diff builds the report-vs-report comparison
import hashlib  # md5 checksum in the JSON payload
from datetime import datetime  # type hint for the created_at argument
from io import BytesIO  # in-memory file, so the PDF never touches disk
from reportlab.pdfgen import canvas  # low-level PDF drawing API
from reportlab.lib.pagesizes import A4  # page dimensions in points
from app.config import Config  # diff threshold and line limit
from app.pool import get_pool  # shared asyncpg pool
from app.memory import _model  # reuse the SAME embedding model as memory, so vectors are comparable


def generate_pdf(title: str, content: str) -> bytes:  # PURPOSE: render a report into PDF bytes, ready to stream back as a download
    buffer = BytesIO()  # write into memory instead of a temp file
    c = canvas.Canvas(buffer, pagesize=A4)  # the drawing surface
    width, height = A4  # ponytail: width is unused, only height drives layout
    c.setFont("Helvetica-Bold", 16)  # title styling
    c.drawString(50, height - 50, title[:80])  # 50pt left margin, truncated so it fits one line
    c.setFont("Helvetica", 10)  # body styling
    y = height - 80  # cursor starts below the title, moves down as we draw
    for line in content.split("\n"):  # walk the report line by line
        for chunk in [line[i:i + 95] for i in range(0, max(len(line), 1), 95)]:  # hard wrap at 95 chars; max(...,1) keeps blank lines alive
            if y < 60:  # ran out of vertical room
                c.showPage()  # start a new page
                y = height - 50  # reset the cursor to the top
            c.drawString(50, y, chunk)  # draw one wrapped chunk
            y -= 14  # line height in points
    c.save()  # finalises the document into the buffer
    buffer.seek(0)  # rewind before reading
    return buffer.read()  # raw PDF bytes


def generate_json_report(topic: str, report: str, report_id: str, created_at: datetime) -> dict:  # PURPOSE: wrap a report as a machine-readable payload with metadata for API consumers
    return {
        "report_id": report_id,  # ties back to the stored row
        "topic": topic,  # what was researched
        "report": report,  # the full text
        "created_at": created_at.isoformat(),  # ISO 8601, JSON-safe
        "word_count": len(report.split()),  # rough length signal for clients
        "checksum": hashlib.md5(report.encode()).hexdigest(),  # integrity check only, md5 is fine here since it guards against corruption, not tampering
    }


async def get_report_diff(config: Config, topic: str) -> str | None:  # PURPOSE: user-facing "what changed" view, comparing the two latest reports on similar topics
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())  # encode blocks, so push it to a thread
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT report, created_at FROM reports
            WHERE 1 - (embedding <=> $1::vector) > $2
            ORDER BY created_at DESC LIMIT 2
            """,  # matched by MEANING, unlike memory.ltm_diff which needs an exact topic string
            str(embedding), config.ltm_diff_threshold,  # looser threshold, default 0.7
        )
        if len(rows) < 2:
            return None  # need two reports to have a diff at all
        old_lines = rows[1]["report"].splitlines(keepends=True)  # index 1 = older
        new_lines = rows[0]["report"].splitlines(keepends=True)  # index 0 = newest
        diff_lines = list(difflib.unified_diff(  # standard +/- unified format
            old_lines, new_lines,
            fromfile=f"previous ({rows[1]['created_at'].date()})",  # dated headers make it readable
            tofile=f"latest ({rows[0]['created_at'].date()})",
            lineterm="",  # lines already carry newlines from keepends=True
        ))
        if not diff_lines:
            return "No significant changes since last report."  # identical text, say so instead of returning empty
        return "\n".join(diff_lines[:config.ltm_diff_limit * 10])  # ponytail: crude line budget, can cut mid-hunk


# Purpose: the presentation layer, turning a finished report into whatever shape the caller
# asked for. generate_pdf renders it as a downloadable A4 document built entirely in memory,
# wrapping long lines and paginating as it goes. generate_json_report returns the same report
# as structured data with metadata (word count, checksum, timestamp) for programmatic clients.
# get_report_diff serves the "what changed since last time" view: it finds the two most recent
# reports on semantically similar topics and returns a unified diff between them, which is why
# it borrows the embedding model from memory.py rather than loading a second copy.
