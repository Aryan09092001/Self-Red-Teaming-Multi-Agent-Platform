import os  # env vars + setting LangChain tracing vars at the bottom
import socket  # gethostname() -> unique consumer name per ECS task
import boto3  # AWS SDK, used here for Secrets Manager
import json  # secret arrives as a JSON string, needs parsing
from functools import lru_cache  # cache the secret fetch so we call AWS once


@lru_cache(maxsize=1)  # one cached result for the whole process lifetime
def _load_secret() -> dict:  # pulls all config from AWS Secrets Manager
    region = os.environ.get("AWS_REGION", "us-east-1")  # env wins, else default region
    client = boto3.client("secretsmanager", region_name=region)  # Secrets Manager client
    response = client.get_secret_value(SecretId="research-agent/config")  # fetch the secret blob
    return json.loads(response["SecretString"])  # parse JSON string into a dict


class Config:  # typed accessor over the secret dict, built once at import/startup
    def __init__(self):  # every attribute below is read from the secret
        data = _load_secret()  # cached, so repeated Config() costs no AWS call

        # AWS
        self.aws_region: str = data.get("AWS_REGION", "us-east-1")  # region for all AWS clients

        # Bedrock Guardrails
        self.bedrock_guardrail_id: str = data["BEDROCK_GUARDRAIL_ID"]  # required: guardrail to apply
        self.bedrock_guardrail_version: str = data["BEDROCK_GUARDRAIL_VERSION"]  # required: pinned version

        # Storage
        self.redis_url: str = data["REDIS_URL"]  # required: cache, sessions, job stream
        self.database_url: str = data["DATABASE_URL"]  # required: Postgres + pgvector
        self.tensorzero_url: str = data["TENSORZERO_URL"]  # required: LLM gateway endpoint

        # Auth
        self.api_key: str = data.get("API_KEY", "")  # empty string = auth disabled

        # LangSmith tracing
        self.langsmith_api_key: str = data.get("LANGSMITH_API_KEY", "")  # empty = tracing off
        self.langchain_project: str = data.get("LANGCHAIN_PROJECT", "research-agent")  # trace project name
        self.langsmith_dataset: str = data.get("LANGSMITH_DATASET", "research-agent-reports")  # eval dataset name

        # Semantic cache
        self.cache_ttl: int = int(data.get("CACHE_TTL", 3600))  # cached answers live 1 hour
        self.cache_similarity_threshold: float = float(data.get("CACHE_SIMILARITY_THRESHOLD", 0.85))  # cosine score needed for a hit

        # Session memory
        self.session_ttl: int = int(data.get("SESSION_TTL", 1800))  # idle session expires after 30 min
        self.session_max_messages: int = int(data.get("SESSION_MAX_MESSAGES", 5))  # keep last N turns only
        self.session_content_truncate: int = int(data.get("SESSION_CONTENT_TRUNCATE", 500))  # chars kept per stored message

        # Long-term memory
        self.ltm_days: int = int(data.get("LTM_DAYS", 7))  # lookback window for recalled reports
        self.ltm_threshold: float = float(data.get("LTM_THRESHOLD", 0.88))  # similarity to treat as the same topic
        self.ltm_diff_threshold: float = float(data.get("LTM_DIFF_THRESHOLD", 0.7))  # looser score for "related, worth diffing"
        self.ltm_diff_limit: int = int(data.get("LTM_DIFF_LIMIT", 5))  # max prior reports pulled for comparison
        self.ivfflat_lists: int = int(data.get("IVFFLAT_LISTS", 100))  # pgvector IVFFlat index partition count

        # Job queue
        self.stream_key: str = data.get("STREAM_KEY", "research:jobs")  # Redis stream holding jobs
        self.consumer_group: str = data.get("CONSUMER_GROUP", "workers")  # shared group so jobs fan out once
        # hostname = unique per ECS task = safe for horizontal scaling
        self.consumer_name: str = data.get("CONSUMER_NAME", socket.gethostname())  # per-worker identity for pending/claim
        self.result_ttl: int = int(data.get("RESULT_TTL", 3600))  # finished job results expire after 1 hour

        # Agent tuning
        self.agent_report_truncate: int = int(data.get("AGENT_REPORT_TRUNCATE", 3000))  # chars of report fed back to the agent
        self.agent_max_iterations: int = int(data.get("AGENT_MAX_ITERATIONS", 2))  # research loop cap, bounds cost

        # Eval tuning
        self.eval_report_truncate: int = int(data.get("EVAL_REPORT_TRUNCATE", 1500))  # chars of report sent to the judge
        self.eval_comment_truncate: int = int(data.get("EVAL_COMMENT_TRUNCATE", 300))  # chars kept from judge feedback

        # LLM retry
        self.llm_max_retries: int = int(data.get("LLM_MAX_RETRIES", 3))  # attempts before giving up on a call
        self.llm_retry_delay: float = float(data.get("LLM_RETRY_DELAY", 1.0))  # base seconds between retries

        # Rate limiting (per IP, per window)
        self.rate_limit_requests: int = int(data.get("RATE_LIMIT_REQUESTS", 10))  # allowed requests per window
        self.rate_limit_window: int = int(data.get("RATE_LIMIT_WINDOW", 60))  # window length in seconds

        # DB connection pool
        self.db_pool_min: int = int(data.get("DB_POOL_MIN", 2))  # connections kept warm
        self.db_pool_max: int = int(data.get("DB_POOL_MAX", 10))  # ceiling, keep under Postgres max_connections

        if self.langsmith_api_key:  # only wire up tracing when a key is present
            os.environ["LANGCHAIN_TRACING_V2"] = "true"  # LangChain reads these from env, not from Config
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key  # auth for the tracing backend
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project  # groups traces under one project
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # hosted LangSmith ingest URL
