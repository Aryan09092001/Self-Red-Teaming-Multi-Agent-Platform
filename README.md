# AI-Security-Project

A production-style **multi-agent research agent** on AWS, built with defence in depth and a red-team harness that continuously attacks it.

The system takes a research topic, runs it through a four-agent LangGraph pipeline, and returns a structured report. Every request is screened by Bedrock Guardrails on the way in and on the way out, every report is scored by four LLM judges, and a separate PyRIT service runs jailbreak, prompt-injection, crescendo and skeleton-key attacks against the live API to prove the guardrails hold.

---

## Table of contents

- [Architecture](#architecture)
- [How a request flows](#how-a-request-flows)
- [Security layers](#security-layers)
- [Red-team harness](#red-team-harness)
- [Evaluation](#evaluation)
- [API reference](#api-reference)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Running locally](#running-locally)
- [Operational notes](#operational-notes)

---

## Architecture

```
                            Internet
                               │
                    ┌──────────▼──────────┐
                    │  Application LB     │  :80 (→ :443 with ACM cert)
                    │                     │  :8001 → PyRIT dashboard
                    └──────┬───────┬──────┘
                           │       │
       ┌───────────────────▼──┐  ┌─▼──────────────────────┐
       │ ECS Fargate: app     │  │ ECS Fargate: pyrit     │
       │ ┌──────────────────┐ │  │                        │
       │ │ FastAPI    :8000 │ │  │ Red-team dashboard     │
       │ │ + worker loop    │ │  │ attacks the ALB ───────┼──┐
       │ ├──────────────────┤ │  │ PyRIT 0.14.0 + SQLite  │  │
       │ │ TensorZero :3000 │ │  └────────────────────────┘  │
       │ │ (sidecar)        │ │                              │
       │ └────────┬─────────┘ │                              │
       └──────────┼───────────┘                              │
                  │                                          │
     ┌────────────┼────────────┬──────────────┐              │
     ▼            ▼            ▼              ▼              │
┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐        │
│ Redis   │ │ Postgres │ │  Bedrock  │ │ OpenAI /   │        │
│ Elasti- │ │ + pgvec- │ │ Guardrails│ │ Groq       │        │
│ Cache   │ │ tor (RDS)│ │           │ │ (via TZ)   │        │
└─────────┘ └──────────┘ └───────────┘ └────────────┘        │
                                                             │
                          attacks flow back in ──────────────┘
```

| Component | Role |
|---|---|
| **FastAPI app** | HTTP API, job queue producer, and the background worker that consumes it |
| **TensorZero** | LLM gateway sidecar on `localhost:3000`; routes to GPT-4o with a Groq Llama 3.1 fallback |
| **LangGraph** | State machine driving the four research agents, with a critic-driven retry loop |
| **Redis (ElastiCache)** | Semantic cache, session memory, rate-limit counters, and the job stream |
| **Postgres + pgvector (RDS)** | Long-term memory: every report stored with a 384-dim topic embedding |
| **Bedrock Guardrails** | Input and output content safety, denied topics, PII blocking |
| **PyRIT dashboard** | Standalone red-team service that attacks the deployed API |
| **LangSmith** | Tracing for every agent step and judge, plus the evaluation dataset |

---

## How a request flows

`POST /research` returns immediately with a `job_id`. Nothing heavy happens on the request thread.

```
POST /research
   ├─ require_api_key         X-API-Key header
   ├─ _rate_limit             per-IP counter in Redis
   ├─ validate_input          Bedrock guardrail on the USER's prompt
   ├─ session_add             record the question as a conversation turn
   └─ push_job                onto the Redis Stream  ──► returns job_id
                                        │
                        background worker picks it up
                                        │
        ┌───────────────────────────────▼──────────────────────────────┐
        │  TIER 1  semantic cache         cosine ≥ 0.85 → reuse answer │
        │  TIER 2  long-term memory       recent near-identical report │
        │  TIER 3  full agent pipeline    the expensive path           │
        └───────────────────────────────┬──────────────────────────────┘
                                        │  (tier 3 only)
        search ──► summarize ──► write ──► verify ──┐
          ▲                                          │
          └──────── critic rejected, retry ◄─────────┘
                    (bounded by AGENT_MAX_ITERATIONS)
                                        │
                    validate_output     Bedrock guardrail on the MODEL's answer
                    cache_set + ltm_store
                    evaluate_report     4 LLM judges, fire-and-forget
                    set_result          client polls GET /result/{job_id}
```

**The three-tier lookup is the cost control.** A repeated question never reaches an LLM: tier 1 catches paraphrases by embedding similarity, tier 2 catches topics researched in the last 7 days. Only genuinely new questions pay for a full pipeline run.

### The four agents

| Agent | Job | Context it receives |
|---|---|---|
| `SearchAgent` | Find 5 key facts and recent developments | Last 4 conversation turns |
| `SummarizeAgent` | Condense findings into structured bullets | Raw search results |
| `WriterAgent` | Draft the report in four fixed sections | Bullets + a *related* past report from long-term memory |
| `CriticAgent` | Verify factual consistency and coherence | The drafted report |

The critic is a real gate: if it replies anything other than `YES`, the orchestrator routes back to `search` and the pipeline runs again, up to `AGENT_MAX_ITERATIONS` times. Quality is bounded by a retry budget rather than an open loop.

---

## Security layers

Seven independent layers, each of which fails closed:

1. **API key auth** — `X-API-Key` header, checked by a FastAPI dependency. Missing and wrong keys return an identical 401, so an attacker learns nothing about which part failed.
2. **Per-IP rate limiting** — atomic Redis counter, default 10 requests per 60s window, returns 429.
3. **Input guardrail** — Bedrock `apply_guardrail` with `source=INPUT` runs *before* the job is even queued.
4. **Output guardrail** — the same guardrail with `source=OUTPUT` runs on the generated report. Blocked reports are never cached and never stored.
5. **Guardrail policy** — HIGH-strength filters for hate, violence, sexual content, insults, misconduct and prompt attack; DENY topics for weapons, illegal activities and self-harm; SSN, credit card and AWS key blocking, with email and phone anonymisation.
6. **Secrets isolation** — nothing is committed. All config lives in AWS Secrets Manager under `research-agent/config`, fetched once at startup and cached in-process.
7. **Network isolation** — Redis and RDS sit in private subnets reachable only from the ECS task security group. VPC endpoints for ECR, S3, Secrets Manager, Bedrock and CloudWatch replace a NAT gateway.

---

## Red-team harness

The PyRIT service is not a test suite that runs in CI — it is a deployed service that attacks the *live* API through the load balancer, exactly as an external attacker would.

| Attack family | Technique | Base risk |
|---|---|---|
| **Jailbreak** | Direct instruction override, DAN-style persona, fictional framing | 8 |
| **XPIA** | Malicious instructions smuggled inside apparently ordinary content | 9 |
| **Crescendo** | Ordered escalation — benign question that ratchets toward a harmful one | 6, +1 per turn |
| **Skeleton Key** | Faked authority ("authorized by the government", "approved by CISO") | 7 |

Reading the results:

- **`blocked: true` means the defence won.** The attack was stopped.
- **A high `risk_score` means an attack got through.** Zero is the target.
- A crescendo chain aborts as soon as the guardrail fires — later turns are meaningless once the escalation is cut off.

Results persist to Redis for 7 days and are rendered by the dashboard at port `8001`. An EventBridge rule reruns the whole suite every Monday at 02:00 UTC.

> Note: the blocked/passed classification is a string heuristic in `pyrit_dashboard/main.py`, not ground truth. A legitimately short answer reads as blocked, which biases the block rate upward. Treat the numbers as a trend, not an audit.

---

## Evaluation

Every research job is automatically scored by four LLM judges running concurrently. Results are pushed to a LangSmith dataset so quality can be tracked over time.

| Judge | Measures | Direction |
|---|---|---|
| `relevance` | Does the report answer the topic asked? | Higher is better |
| `completeness` | Are all four required sections present? | Higher is better |
| `hallucination_risk` | Fabricated stats, impossible dates, contradictions | **Lower is better** |
| `overall_quality` | Depth, accuracy, clarity, structure, usefulness | Higher is better |

**`hallucination_risk` uses an inverted scale.** Do not average the four scores together without accounting for that.

`POST /run-evaluation` reruns the full pipeline over recent real topics as a regression harness. If no topics are supplied it pulls the most recent distinct topics from the database, so the regression set reflects what users actually asked.

---

## API reference

All endpoints except `/` and `/health` require `X-API-Key`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Single-page frontend |
| `GET` | `/health` | Liveness probe for the ALB (checks Redis) |
| `POST` | `/research` | Queue a research job → `{job_id, session_id}` |
| `GET` | `/result/{job_id}` | Poll for the result, or `{"status": "pending"}` |
| `GET` | `/result/{job_id}/pdf` | Download any finished report as a PDF |
| `GET` | `/session/{session_id}` | Replay a conversation's recent turns |
| `GET` | `/diff/{topic}` | Unified diff between the two latest reports on a topic |
| `GET` | `/stats` | Redis key counts, memory, uptime, wired-up backends |
| `GET` | `/evaluate/{job_id}` | Re-score a finished report on demand |
| `POST` | `/run-evaluation` | Kick off a batch regression run in the background |

Example:

```bash
# submit
curl -X POST https://<alb-dns>/research \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"topic": "state of solid-state batteries", "output_format": "json"}'
# → {"job_id": "…", "session_id": "…"}

# poll
curl https://<alb-dns>/result/<job_id> -H "X-API-Key: $API_KEY"
```

`output_format` accepts `text` (default), `pdf` (base64 in the result payload) or `json` (structured, with word count and checksum).

---

## Project layout

```
app/
  main.py        FastAPI app, routes, worker loop, job pipeline
  agents.py      LangGraph state machine and the four research agents
  memory.py      Short-term (Redis) + long-term (pgvector) memory
  cache.py       Semantic cache by embedding similarity
  guardrails.py  Bedrock input/output validation
  eval.py        Four LLM judges + LangSmith logging + batch harness
  queue.py       Redis Streams job queue with consumer groups
  output.py      PDF, JSON and diff rendering
  config.py      Typed config loaded once from Secrets Manager
  auth.py        API key dependency
  pool.py        Shared asyncpg connection pool
  retry.py       Exponential-backoff helper for flaky async calls
  Dockerfile

pyrit_dashboard/
  main.py        Red-team service: attack sets, runner, dashboard UI
  Dockerfile

tensorzero/
  tensorzero.toml            Model routing and function definitions
  templates/*.minijinja      System prompts for each function
  Dockerfile

terraform/main.tf            All infrastructure, single file
.github/workflows/deploy.yml Build → ECR → ECS, with rollback
bootstrap.sh / bootstrap.bat Creates the Terraform state backend
index.html                   Frontend served at /
```

Every source file carries inline comments and a purpose block at the bottom explaining what it does and why.

---

## Configuration

All settings live in one Secrets Manager secret, `research-agent/config`, and are read once at startup by `app/config.py`. Required keys raise on boot; everything else falls back to a documented default.

**Required:** `BEDROCK_GUARDRAIL_ID`, `BEDROCK_GUARDRAIL_VERSION`, `REDIS_URL`, `DATABASE_URL`, `TENSORZERO_URL`.

**Notable tunables:**

| Key | Default | Effect |
|---|---|---|
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | How alike two questions must be to reuse an answer |
| `LTM_THRESHOLD` | `0.88` | Similarity to treat a past report as the same topic |
| `LTM_DAYS` | `7` | Lookback window for long-term memory reuse |
| `AGENT_MAX_ITERATIONS` | `2` | Critic retry budget — the main cost lever |
| `RATE_LIMIT_REQUESTS` / `_WINDOW` | `10` / `60` | Per-IP throttle |
| `SESSION_TTL` / `SESSION_MAX_MESSAGES` | `1800` / `5` | Conversation memory window |
| `API_KEY` | `""` | **Empty disables auth entirely** — never leave empty in production |

Terraform writes this secret with `REPLACE_ME` placeholders for `OPENAI_API_KEY`, `GROQ_API_KEY` and `LANGSMITH_API_KEY`. Fill those in after the first apply.

---

## Deployment

**One-time bootstrap** — creates the versioned, encrypted, private S3 bucket and DynamoDB lock table that hold Terraform state:

```bash
./bootstrap.sh us-east-1      # or bootstrap.bat on Windows
```

**Infrastructure:**

```bash
cd terraform
terraform init
terraform apply -var="app_image=placeholder" -var="pyrit_image=placeholder"
```

Then replace the three `REPLACE_ME` values in the `research-agent/config` secret with real API keys.

**Application** — push to `main`. GitHub Actions builds all three images, pushes them to ECR, registers new ECS task definitions, and updates both services. If the app service fails to stabilise, the workflow **automatically rolls both services back** to the previously running task definition.

Requires repo secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

**What Terraform provisions:** VPC across 2 AZs with public and private subnets, six VPC endpoints (no NAT gateway), an ALB, an ECS Fargate cluster with CPU-target auto-scaling from 1 to 5 tasks, RDS Postgres 15.8 with 7-day backups, ElastiCache Redis 7.1, the Bedrock guardrail and its version, three ECR repositories with scan-on-push, CloudWatch log groups, and the weekly red-team EventBridge rule.

---

## Running locally

There is no compose file in the repo. To run locally you need Redis, Postgres with the `pgvector` extension, and a TensorZero gateway reachable on `TENSORZERO_URL`, plus AWS credentials that can read the secret and call `bedrock:ApplyGuardrail`.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`app/config.py` reads from Secrets Manager unconditionally — there is no local `.env` path — so a local run still needs access to the `research-agent/config` secret.

The dashboard runs separately:

```bash
pip install -r pyrit_dashboard/requirements.txt
TARGET_URL=http://localhost:8000 uvicorn pyrit_dashboard.main:app --port 8001
```

---

## Operational notes

Things worth knowing before this goes anywhere near real traffic:

- **CORS is wide open.** `allow_origins=["*"]` in `app/main.py`. Since auth is a header key rather than a cookie this is not a CSRF hole, but any page can call the API if it holds the key. Restrict to the real frontend origin before production.
- **The worker shares a process with the API.** `_worker_loop` is started as an asyncio task inside the app container. Fine at current scale; split it into its own service if job volume grows.
- **`/health` only checks Redis.** Postgres could be down and the ALB would still see a healthy target.
- **Unknown and expired job IDs both return `{"status": "pending"}`**, so a client polling a stale ID waits forever.
- **The semantic cache scans all embedding keys per lookup.** Linear in cache size — fine for hundreds of entries, not for hundreds of thousands.
- **Cache keys use Python's `hash()`**, which is randomised per process. Two workers writing the same query produce two entries. Reads still work (they go through the similarity scan), but storage duplicates.

Each of these is marked with a `ponytail:` comment at the exact line in the source, so `grep -rn "ponytail:" .` gives the full ledger of known shortcuts.
