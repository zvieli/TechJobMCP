# Adaptive Per-Source Timeouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate premature aggregation timeouts by replacing the rigid uniform 12.0s ceiling with source-specific empirical timeout budgets (ranging from 10.0s for lightweight REST APIs up to 25.0s for multi-hub HTML scrapers) and dynamically calculating the aggregator ceiling as `max(active_source_timeouts) + 2.0s`.

**Architecture:**
1. **Contract & Metadata:** Extend `BaseJobSource` and `SourceMetadata` with a `timeout: float` field and a `get_timeout()` method that allows per-source environment variable overrides (`SOURCE_TIMEOUT_<NAME>`).
2. **Empirical Source Budgets:** Configure tailored timeout defaults for all 11 sources based on network depth, number of remote boards, and scraping pacing.
3. **Adaptive Aggregator:** Update `JobAggregator.fetch_all_jobs()` so each source task is isolated in `asyncio.wait_for(src.fetch_jobs(...), timeout=effective_timeout)`. The overall ceiling is calculated dynamically as $\max_{s \in \text{active}} (\tau_s) + 2.0\text{s}$. If an explicit `source_timeout` is passed to `JobAggregator`, it acts as an override cap for strict backward compatibility.
4. **Server Lifespan & Configuration:** Increase background cache warmup timeout in `main.py` (`_WARMUP_TIMEOUT_SECONDS = 60.0`), document source overrides in `.env.example`, and remove the hardcoded `SOURCE_TIMEOUT_SECONDS=12.0` constraint from `.env` and `docker-compose.yml`.

**Tech Stack:** Python 3.12, asyncio, httpx, FastMCP, Docker Compose, pytest, pytest-asyncio.

---

## Empirical Benchmark & Time Cap Research Summary

Based on live single-source latency benchmarks and concurrent Docker network profiling:

| Source ID | Category | Architecture / Upstream | Measured Isolated | Measured Concurrent | Proposed Timeout Cap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`eightfold`** | Public | 3 enterprise search APIs (Micron, PayPal, NVIDIA) | 1.85s | ~5.3s | **10.0s** |
| **`alljobs`** | Public | Mobile search feed (deprecated/fallback) | 1.36s | ~2.5s | **10.0s** |
| **`hiremetech`** | Authenticated | REST API with session auth | 1.88s | ~3.7s | **12.0s** |
| **`direct_tech`** | Enterprise | 4 major career APIs (Google, Apple, Amazon, IBM) | 2.14s | ~8.2s | **15.0s** |
| **`workday`** | Enterprise | 5 CXS enterprise portals (Cisco, Philips, NVIDIA, Intel, Autodesk) | 4.22s | ~7.8s | **15.0s** |
| **`lever`** | Public | 14 public company boards via Lever API | 2.18s | ~12.0s | **15.0s** |
| **`comeet`** | Public | 11 Israeli company boards via Comeet API | 7.70s | ~12.5s | **18.0s** |
| **`jobify`** | Public | HTML scraper with JSON-LD parsing across seed URLs | 4.41s | ~12.0s | **18.0s** |
| **`greenhouse`** | Public | 7 Israeli company boards with full HTML description payloads (`content=true`) | 10.40s | ~14.0s | **20.0s** |
| **`linkedin`** | Authenticated | Multi-keyword guest search with pagination and 429 retry backoff | 0.83s | ~12.0s+ | **20.0s** |
| **`gotfriends`** | Public | 10 category hubs, semaphore limit (3), 0.5s-0.75s request pacing | 12.00s | ~18.0s | **25.0s** |

**Aggregator Global Ceiling:** Dynamic $\max(\tau_i) + 2.0\text{s} = 25.0\text{s} + 2.0\text{s} = \mathbf{27.0\text{s}}$ when all sources are active.

---

## Global Constraints

- Do NOT hardcode any candidate PII in code, docstrings, or tests.
- Maintain 100% backward compatibility with existing tests, signatures, and default behaviors:
  - `JobAggregator(source_timeout=1.5)` must still act as an explicit override cap across all sources.
  - `DEFAULT_SOURCE_TIMEOUT` must remain an exported symbol in `job_mcp.sources.aggregator` and `job_mcp.sources`.
- All 897 existing unit and integration tests must remain green.
- Every source must gracefully isolate its own failure or timeout without blocking or aborting other sources in concurrent aggregation.

---

## File Structure

```
job_mcp/
  sources/
    base.py                     # Add default_timeout to SourceMetadata, timeout & get_timeout() to BaseJobSource
    contracts.py                # Add default_timeout attribute to IJobSource protocol
    aggregator.py               # Update JobAggregator to compute adaptive timeouts per source and dynamic ceiling
    authenticated/
      hiremetech.py             # timeout = 12.0
      linkedin.py               # timeout = 20.0
    enterprise/
      direct_tech.py            # timeout = 15.0
      workday.py                # timeout = 15.0
    public/
      alljobs.py                # timeout = 10.0
      comeet.py                 # timeout = 18.0
      eightfold.py              # timeout = 10.0
      gotfriends.py             # timeout = 25.0
      greenhouse.py             # timeout = 20.0
      jobify.py                 # timeout = 18.0
      lever.py                  # timeout = 15.0
  main.py                       # Update _WARMUP_TIMEOUT_SECONDS and cache warmup timeout handling
.env                            # Remove rigid SOURCE_TIMEOUT_SECONDS=12.0
.env.example                    # Document per-source and global timeout configuration
tests/
  test_aggregator.py            # Add tests for adaptive per-source timeouts, env overrides, and dynamic ceilings
  test_sources.py               # Verify source timeouts and get_timeout() behavior
```

---

## Task Decomposition

### Task 1: Source Base Contract & Metadata Extension

**Files:**
- Modify: `job_mcp/sources/contracts.py:25-50`
- Modify: `job_mcp/sources/base.py:20-100`
- Test: `tests/test_sources.py`

**Interfaces:**
- `SourceMetadata(..., default_timeout: float = 15.0)`
- `IJobSource` protocol: includes `timeout: float` and `get_timeout() -> float`
- `BaseJobSource`:
  - `timeout: float = 15.0`
  - `get_timeout(self) -> float`: checks `SOURCE_TIMEOUT_<SOURCE_ID>`, falls back to `self.timeout`.
  - `get_metadata(self) -> SourceMetadata`: populates `default_timeout=self.get_timeout()`.

- [ ] **Step 1: Write failing unit test for `get_timeout()` and env overrides**

In `tests/test_sources.py`:
```python
def test_base_source_get_timeout_default_and_env_override(monkeypatch):
    from job_mcp.sources.base import BaseJobSource

    class DummySource(BaseJobSource):
        source_id = "dummy_src"
        display_name = "Dummy"
        timeout = 14.5

        async def fetch_jobs(self, preferences=None, limit=50):
            return []

    src = DummySource()
    assert src.get_timeout() == 14.5
    assert src.get_metadata().default_timeout == 14.5

    # Test environment variable override: SOURCE_TIMEOUT_DUMMY_SRC
    monkeypatch.setenv("SOURCE_TIMEOUT_DUMMY_SRC", "22.0")
    assert src.get_timeout() == 22.0
    assert src.get_metadata().default_timeout == 22.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_sources.py -k test_base_source_get_timeout_default_and_env_override -v`
Expected: FAIL with AttributeError (`default_timeout` or `get_timeout` not defined).

- [ ] **Step 3: Implement `SourceMetadata` and `BaseJobSource` enhancements**

In `job_mcp/sources/contracts.py`:
Add `default_timeout: float = 15.0` to `SourceMetadata` dataclass, and add `timeout: float` and `get_timeout() -> float` to `IJobSource`.

In `job_mcp/sources/base.py`:
```python
@dataclass
class SourceMetadata:
    source_id: str
    display_name: str
    description: str
    category: SourceCategory
    requires_auth: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False
    is_healthy: bool = True
    default_timeout: float = 15.0


class BaseJobSource(ABC):
    source_id: str = "base"
    display_name: str = "Base Source"
    description: str = "Base job source interface"
    category: SourceCategory = SourceCategory.PUBLIC
    requires_auth: bool = False
    supports_bookmarks: bool = False
    supports_auto_apply: bool = False
    timeout: float = 15.0

    def get_timeout(self) -> float:
        """Get effective timeout in seconds for this source, factoring in environment overrides."""
        env_key = f"SOURCE_TIMEOUT_{self.source_id.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            try:
                return float(env_val.strip())
            except ValueError:
                pass
        return getattr(self, "timeout", 15.0)

    def get_metadata(self) -> SourceMetadata:
        """Return structured metadata describing this source."""
        return SourceMetadata(
            source_id=self.source_id,
            display_name=self.display_name,
            description=self.description,
            category=self.category,
            requires_auth=self.requires_auth,
            supports_bookmarks=self.supports_bookmarks,
            supports_auto_apply=self.supports_auto_apply,
            default_timeout=self.get_timeout(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_sources.py -k test_base_source_get_timeout_default_and_env_override -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add job_mcp/sources/base.py job_mcp/sources/contracts.py tests/test_sources.py
git commit -m "feat(sources): add default_timeout contract and per-source env override to BaseJobSource"
```

---

### Task 2: Configure Empirical Source-Specific Timeouts Across All 11 Sources

**Files:**
- Modify:
  - `job_mcp/sources/authenticated/hiremetech.py` (class attribute `timeout: float = 12.0`)
  - `job_mcp/sources/authenticated/linkedin.py` (class attribute `timeout: float = 20.0`)
  - `job_mcp/sources/enterprise/direct_tech.py` (class attribute `timeout: float = 15.0`)
  - `job_mcp/sources/enterprise/workday.py` (class attribute `timeout: float = 15.0`)
  - `job_mcp/sources/public/alljobs.py` (class attribute `timeout: float = 10.0`)
  - `job_mcp/sources/public/comeet.py` (class attribute `timeout: float = 18.0`)
  - `job_mcp/sources/public/eightfold.py` (class attribute `timeout: float = 10.0`)
  - `job_mcp/sources/public/gotfriends.py` (class attribute `timeout: float = 25.0`)
  - `job_mcp/sources/public/greenhouse.py` (class attribute `timeout: float = 20.0`)
  - `job_mcp/sources/public/jobify.py` (class attribute `timeout: float = 18.0`)
  - `job_mcp/sources/public/lever.py` (class attribute `timeout: float = 15.0`)
- Test: `tests/test_sources.py`

- [ ] **Step 1: Write unit test validating timeout baseline for all registered sources**

In `tests/test_sources.py`:
```python
def test_registered_sources_have_tailored_timeouts():
    from job_mcp.sources.registry import create_default_registry

    reg = create_default_registry()
    sources = {s.source_id: s.get_timeout() for s in reg.get_all()}

    expected_timeouts = {
        "eightfold": 10.0,
        "hiremetech": 12.0,
        "direct_tech": 15.0,
        "workday": 15.0,
        "lever": 15.0,
        "comeet": 18.0,
        "jobify": 18.0,
        "greenhouse": 20.0,
        "linkedin": 20.0,
        "gotfriends": 25.0,
    }
    for sid, expected_t in expected_timeouts.items():
        if sid in sources:
            assert sources[sid] == expected_t, f"Source {sid} timeout mismatch: {sources[sid]} != {expected_t}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_sources.py -k test_registered_sources_have_tailored_timeouts -v`
Expected: FAIL (as classes currently have default 15.0).

- [ ] **Step 3: Update timeout attributes in all 11 source classes**

Set `timeout: float = <value>` on each source class:
- `HireMeTechSource`: `timeout = 12.0`
- `LinkedInSource`: `timeout = 20.0`
- `DirectTechSource`: `timeout = 15.0`
- `WorkdaySource`: `timeout = 15.0`
- `AllJobsSource`: `timeout = 10.0`
- `ComeetSource`: `timeout = 18.0`
- `EightfoldAISource`: `timeout = 10.0`
- `GotFriendsSource`: `timeout = 25.0`
- `GreenhouseSource`: `timeout = 20.0`
- `JobifySource`: `timeout = 18.0`
- `LeverSource`: `timeout = 15.0`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_sources.py -k test_registered_sources_have_tailored_timeouts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add job_mcp/sources/ tests/test_sources.py
git commit -m "feat(sources): configure tailored empirical timeout caps for all 11 sources"
```

---

### Task 3: Dynamic Adaptive Aggregator Timeout in JobAggregator

**Files:**
- Modify: `job_mcp/sources/aggregator.py:20-60, 190-235`
- Test: `tests/test_aggregator.py`

**Interfaces:**
- `DEFAULT_SOURCE_TIMEOUT: float = float(os.getenv("SOURCE_TIMEOUT_SECONDS", "25.0"))` (raised from 12.0 to 25.0 as sensible fallback default).
- `JobAggregator.__init__(..., source_timeout: Optional[float] = None)`
  - If `source_timeout` is passed explicitly (e.g. `JobAggregator(source_timeout=1.5)`), it acts as an explicit override cap on all sources.
  - If `source_timeout is None`, adaptive per-source timeouts are used.
  - Property `aggregator.source_timeout`: returns explicit override if set, else `aggregator.get_max_timeout()`.
- Method `get_source_timeout(self, source: IJobSource) -> float`:
  - Returns `min(source.get_timeout(), self._explicit_timeout)` if `self._explicit_timeout is not None` else `source.get_timeout()`.
- Method `get_max_timeout(self, sources: Optional[list[IJobSource]] = None) -> float`:
  - Returns $\max_{s} (\text{get\_source\_timeout}(s))$.
- In `fetch_all_jobs()`:
  - For each `src` in `active_sources`, timeout is `src_timeout = self.get_source_timeout(src)`.
  - Aggregator ceiling is `max_timeout = self.get_max_timeout(active_sources) + 2.0`.
  - `_fetch_with_timeout(src)` uses `src_timeout`.
  - Log line: `"Fetching jobs concurrently from %d source(s) with adaptive timeouts (source range: %.1fs - %.1fs, ceiling: %.1fs)"`

- [ ] **Step 1: Write failing tests in `tests/test_aggregator.py` for adaptive timeouts and dynamic ceiling**

In `tests/test_aggregator.py`:
```python
async def test_aggregator_adaptive_per_source_timeouts() -> None:
    class FastCustomSource(MockSource):
        timeout = 2.0

    class SlowCustomSource(MockSource):
        timeout = 8.0

    src_fast = FastCustomSource("fast", delay=0.1)
    src_slow = SlowCustomSource("slow", delay=0.1)

    reg = SourceRegistry()
    reg.register(src_fast)
    reg.register(src_slow)

    agg = JobAggregator(registry=reg)  # No explicit source_timeout
    assert agg.get_source_timeout(src_fast) == 2.0
    assert agg.get_source_timeout(src_slow) == 8.0
    assert agg.get_max_timeout([src_fast, src_slow]) == 8.0

async def test_aggregator_explicit_override_still_clamps_all_sources() -> None:
    class CustomSource(MockSource):
        timeout = 20.0

    src = CustomSource("custom", delay=0.1)
    reg = SourceRegistry()
    reg.register(src)

    agg = JobAggregator(registry=reg, source_timeout=1.5)
    # Explicit source_timeout overrides/clamps
    assert agg.get_source_timeout(src) == 1.5
    assert agg.source_timeout == 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_aggregator.py -k "test_aggregator_adaptive_per_source_timeouts or test_aggregator_explicit_override_still_clamps_all_sources" -v`
Expected: FAIL.

- [ ] **Step 3: Implement adaptive timeout calculation and backward compatibility in `JobAggregator`**

In `job_mcp/sources/aggregator.py`:
Update `DEFAULT_SOURCE_TIMEOUT`, constructor, `get_source_timeout`, `get_max_timeout`, and `_fetch_with_timeout`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_aggregator.py -v`
Expected: PASS all aggregator tests (including existing timeout tests).

- [ ] **Step 5: Commit**

```bash
git add job_mcp/sources/aggregator.py tests/test_aggregator.py
git commit -m "feat(aggregator): implement adaptive per-source timeouts and dynamic ceiling"
```

---

### Task 4: Lifespan Warmup, Docker & Environment Configuration

**Files:**
- Modify: `job_mcp/main.py:150-170` (`_WARMUP_TIMEOUT_SECONDS = 60.0`, dynamic ceiling calculation)
- Modify: `.env.example`
- Modify: `.env`
- Modify: `docker-compose.yml`
- Test: `tests/test_main_startup.py` or `tests/test_tools.py`

- [ ] **Step 1: Update `job_mcp/main.py`**
  - Increase `_WARMUP_TIMEOUT_SECONDS` to `60.0` so background warmup can comfortably finish all 10 sources even under high network latency.
  - In `_warm_cache`, compute dynamic timeout:
    `max_t = agg.get_max_timeout() + 5.0`
    `await asyncio.wait_for(..., timeout=max(_WARMUP_TIMEOUT_SECONDS, max_t))`
- [ ] **Step 2: Update `.env.example` and `.env`**
  - In `.env.example` and `.env`:
    Remove or comment out the restrictive `SOURCE_TIMEOUT_SECONDS=12.0`.
    Document per-source timeout override variables:
    ```bash
    # Aggregator & Source Timeouts (Seconds)
    # SOURCE_TIMEOUT_SECONDS=              # Optional global clamp. Leave empty to use adaptive timeouts.
    # SOURCE_TIMEOUT_GOTFRIENDS=25.0
    # SOURCE_TIMEOUT_GREENHOUSE=20.0
    # SOURCE_TIMEOUT_LINKEDIN=20.0
    # SOURCE_TIMEOUT_COMEET=18.0
    # SOURCE_TIMEOUT_JOBIFY=18.0
    # SOURCE_TIMEOUT_WORKDAY=15.0
    # SOURCE_TIMEOUT_DIRECT_TECH=15.0
    # SOURCE_TIMEOUT_LEVER=15.0
    # SOURCE_TIMEOUT_HIREMETECH=12.0
    # SOURCE_TIMEOUT_EIGHTFOLD=10.0
    # SOURCE_TIMEOUT_ALLJOBS=10.0
    ```
- [ ] **Step 3: Update `docker-compose.yml`**
  - Ensure `techjob-mcp` service does not inject a hardcoded `SOURCE_TIMEOUT_SECONDS: "12.0"`.
- [ ] **Step 4: Run test suite**
  - Run `.venv/bin/pytest tests/test_tools.py tests/test_aggregator.py -v`.
  - Verify PASS.
- [ ] **Step 5: Commit**
```bash
git add job_mcp/main.py .env.example .env docker-compose.yml
git commit -m "chore(config): configure adaptive source timeouts and 60s background warmup"
```

---

### Task 5: Live Docker Integration & End-to-End Verification

**Files:**
- None (Operational verification)

- [ ] **Step 1: Re-run full test suite**
  - Command: `.venv/bin/pytest tests/ -q`
  - Expected: 897+ passed, 0 failures.
- [ ] **Step 2: Run live aggregator benchmark script across all active sources concurrently**
  - Command: `.venv/bin/python scripts/run_benchmark.py` (or inline python invocation).
  - Verify that Comeet, Greenhouse, GotFriends, Jobify, and LinkedIn all complete without timing out.
- [ ] **Step 3: Restart Docker stack and inspect container logs**
  - Command: `docker compose down && docker compose up -d && sleep 10 && docker compose logs techjob-mcp`
  - Verify that background cache warmup logs show:
    `Fetching jobs concurrently from 10 source(s) with adaptive timeouts`
    and **ZERO** `"Source '...' timed out after 12.0s."` warnings!

---

## Self-Review Checklist
1. **Spec Coverage:**
   - Researched each source's empirical latency and time cap? Yes, detailed table included.
   - Applied software design best practices (SRP, OCP, dynamic ceiling)? Yes, BaseJobSource metadata contract, IJobSource protocol, per-source env overrides, dynamic aggregator ceiling.
   - Set server aggregator ceiling to max from all of them? Yes, `max(active_source_timeouts) + 2.0s`.
2. **Backward Compatibility:**
   - Existing tests checking `JobAggregator(source_timeout=1.5)` or `DEFAULT_SOURCE_TIMEOUT`? Yes, fully supported and tested.
3. **No Placeholders:**
   - All steps contain exact paths, commands, and code logic.
