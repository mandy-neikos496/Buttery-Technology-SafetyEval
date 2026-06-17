"""
Safety & Bias Evaluation Suite — data structures.

Pydantic v2 models for the unified ethics eval runner.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# Axes and benchmarks

class EthicsAxis(str, Enum):
    BIAS = "bias"
    STEREOTYPE = "stereotype"
    TOXICITY = "toxicity"
    HARM = "harm"
    FACTUALITY = "factuality"


class EvalBenchmark(BaseModel):
    """Metadata about one benchmark dataset."""
    name: str                       # benchmark names
    axis: EthicsAxis
    description: str
    source_url: str
    license: str
    sensitivity: str                # "low" / "medium" / "high" — content sensitivity warning

# Items and expected behaviors

class ExpectedBehavior(str, Enum):
    """What we hope the model does."""
    REFUSE = "refuse"                       # for harm prompts
    SELECT_LABEL = "select_label"           # for classification benchmarks
    AVOID_STEREOTYPE = "avoid_stereotype"   # for stereotype tasks (prefer anti-stereotype completion)
    ANSWER_FACTUALLY = "answer_factually"   # for factuality tasks


class EvalItem(BaseModel):
    """One prompt to evaluate."""
    id: str
    benchmark: str
    axis: EthicsAxis
    input_text: str
    expected_behavior: ExpectedBehavior
    # Labels for classification-style benchmarks
    labels: Optional[list[str]] = None
    correct_label: Optional[str] = None
    # For stereotype tasks: pairs of (stereotype, anti-stereotype, unrelated)
    completion_options: Optional[dict[str, str]] = None
    # Demographic / topical metadata for slicing analysis
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Sensitivity warning for this specific item - useful for filtered display
    sensitivity: str = "low"

# LLM configuration

class LLMConfig(BaseModel):
    provider: str
    model_id: str
    base_url: str
    api_key_env: str
    max_tokens: int = 512
    temperature: float = 0.0

# Responses and scoring

class ResponseOutcome(str, Enum):
    """Categorized outcome of one model response."""
    REFUSED = "refused"                     # model declined to comply
    COMPLIED = "complied"                   # model produced the requested (harmful) content
    SELECTED_CORRECTLY = "selected_correctly"
    SELECTED_BIASED = "selected_biased"     # picked a biased answer
    SELECTED_UNKNOWN = "selected_unknown"   # admitted not knowing (often correct on BBQ)
    GENERATED_TOXIC = "generated_toxic"
    GENERATED_NON_TOXIC = "generated_non_toxic"
    ANSWERED_TRUTHFULLY = "answered_truthfully"
    ANSWERED_FALSELY = "answered_falsely"
    ERROR = "error"
    UNCLEAR = "unclear"                     # couldn't categorize


class EthicsResponse(BaseModel):
    """One model's response to one item — raw + parsed signals."""
    item_id: str
    model_id: str
    raw_text: str
    outcome: ResponseOutcome
    classifier_score: Optional[float] = None  # if a classifier (Detoxify, LLM-judge) rated it
    classifier_name: Optional[str] = None
    latency_ms: int
    tokens_in: int
    tokens_out: int
    error: Optional[str] = None
    is_safe_to_display: bool = True
    timestamp: datetime


class EthicsResult(BaseModel):
    """Scoring outcome for one item."""
    item_id: str
    benchmark: str
    axis: EthicsAxis
    model_id: str
    is_desired_behavior: bool       # True if model did what we hoped
    severity: str                   # "ok" / "minor" / "moderate" / "serious"
    response: EthicsResponse

# Aggregate reports

class AxisReport(BaseModel):
    """Aggregate score for one (model × axis)."""
    model_id: str
    axis: EthicsAxis
    benchmarks_run: list[str]
    n_items: int
    n_desired: int                  # count of "did what we hoped"
    desired_rate: float             # n_desired / n_items
    # For harm: refusal_rate. For bias: anti-bias_rate.
    serious_failure_count: int      # items that were "serious" severity
    notes: str = ""


class ModelReportCard(BaseModel):
    """All axes for one model."""
    model_id: str
    axes: list[AxisReport]
    overall_safety_score: float   
    timestamp: datetime


class ReportGrid(BaseModel):
    """The headline comparison: rows are models, columns are axes."""
    generated_at: datetime
    cards: list[ModelReportCard]
    methodology_summary: str
    known_limitations: list[str]

    def to_markdown(self) -> str:
        raise NotImplementedError

# Outcome classifier protocol

class OutcomeClassifier:
    """
    Abstract classifier — turns a raw model response into a structured outcome.

    Implementations:
      - RefusalClassifier — detects refusal in harm-benchmark responses (regex + LLM-judge)
      - LabelExtractor — pulls a label choice from open generation (for BBQ etc.)
      - ToxicityClassifier — wraps Detoxify or an LLM-judge for toxicity scoring
      - TruthfulnessClassifier — for TruthfulQA, uses the canonical scoring fns

    Each classifier returns (ResponseOutcome, optional_score).
    """

    name: str

    def classify(self, item: EvalItem, raw_text: str) -> tuple[ResponseOutcome, Optional[float]]:
        raise NotImplementedError
