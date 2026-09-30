"""Milestone 4 deterministic verification of the Docker Compose topology.

These tests assert semantic invariants of ``docker-compose.yml`` — required
service set, Ollama healthcheck semantics, model-preparation dependency
ordering, observability wiring, and the pinned MCP environment values. They
never snapshot the rendered document, so benign refactors keep passing while
topology regressions fail loudly.

Everything here is offline, zero-cost and CPU-only: no image builds, no model
downloads and no paid LLM APIs. Docker-backed checks skip when the Compose CLI
is unavailable rather than pretending runtime health was proven.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker-compose.yml"

REQUIRED_SERVICES = {
    "techjob-mcp",
    "prometheus",
    "grafana",
    "ollama",
    "ollama-model-pull",
}

# Milestone 4 contract: declared literally in the service definition, so the
# topology does not depend on a developer-local .env for these values.
REQUIRED_MCP_ENV = {
    "SOURCE_TIMEOUT_SECONDS": "15.0",
    "ENABLE_INT8_QUANTIZATION": "true",
    "MAX_NEURAL_EVAL": "15",
    "TORCH_NUM_THREADS": "4",
}

# A model name inside the readiness probe is the exact defect class: the daemon
# can never become healthy on a volume that does not hold that specific model.
MODEL_TOKENS = ("OLLAMA_MODEL", "llama3.2", "qwen")


@pytest.fixture(scope="module")
def compose() -> dict:
    document = yaml.safe_load(COMPOSE_FILE.read_text())
    assert isinstance(document, dict), "docker-compose.yml must be a mapping"
    return document


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    assert isinstance(compose["services"], dict)
    return compose["services"]


def _env_map(service: dict) -> dict[str, str]:
    """Normalise Compose list-form and map-form ``environment`` into one dict."""
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    merged: dict[str, str] = {}
    for entry in environment:
        key, _, value = str(entry).partition("=")
        merged[key] = value
    return merged


def _depends_on(service: dict) -> dict[str, str]:
    """Normalise ``depends_on`` into ``{service: condition}`` ("" if implicit)."""
    dependencies = service.get("depends_on") or {}
    if isinstance(dependencies, list):
        return {str(name): "" for name in dependencies}
    return {
        str(name): str((spec or {}).get("condition", "")) for name, spec in dependencies.items()
    }


def _healthcheck_text(service: dict) -> str:
    test = service["healthcheck"]["test"]
    parts = test if isinstance(test, list) else [test]
    return " ".join(str(part) for part in parts)


def _pull_script(service: dict) -> str:
    """Raw model-pull script text, exactly as Compose stores it.

    Comment handling is left to :func:`_shell_commands`, which is quote-aware and
    so cannot be fooled by a ``#`` inside a quoted string.
    """
    command = service.get("command") or []
    if isinstance(command, str):
        return command
    return "\n".join(str(part) for part in command)


# ---------------------------------------------------------------------------
# Minimal POSIX-shell statement parser
#
# Loose text matching is not proof: a line like
# `echo 'timeout --kill-after=10s "$PULL_DEADLINE" ollama pull "$TARGET_MODEL"'`
# contains the exact invocation as a string without ever running it, and an
# inline comment can carry the same text. Parsing into real command token lists
# means an assertion is satisfied only by an executed command. The splitter is
# quote-aware, so comment markers inside quotes are not stripped.
# ---------------------------------------------------------------------------

_SHELL_CONTROL_WORDS = {
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "while",
    "until",
    "for",
    "in",
    "do",
    "done",
    "{",
    "}",
    "(",
    ")",
}
_REDIRECT_RE = re.compile(r"^(?:\d*)[<>][&]?")


def _split_shell_lines(script: str) -> list[str]:
    """Split script text into lines with unquoted trailing comments removed."""
    lines: list[str] = []
    for raw in script.splitlines():
        line = raw.strip()
        quote = ""
        index = 0
        while index < len(line):
            char = line[index]
            if quote:
                if char == quote:
                    quote = ""
            elif char in {"'", '"'}:
                quote = char
            elif char == "#" and (index == 0 or line[index - 1] in " \t"):
                line = line[:index]
                break
            index += 1
        if line.strip():
            lines.append(line.strip())
    return lines


def _split_segments(line: str) -> list[str]:
    """Split one line into simple commands on unquoted ``;``, ``&&`` and ``||``.

    A ``]`` or ``>``, closing a test bracket or ending a redirection, may sit
    directly against the separator, so those are spaced first.
    """
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == ";":
            segments.append("".join(current))
            current = []
            index += 1
            continue
        if line[index : index + 2] in {"&&", "||"}:
            segments.append("".join(current))
            current = []
            index += 2
            continue
        current.append(char)
        index += 1
    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def _normalise_token(token: str) -> str:
    """Collapse Compose escaping and braced expansion so both forms compare equal.

    The raw YAML writes `$${VAR}` and the rendered/container form writes `${VAR}`
    or `$VAR`; all three denote the same shell expansion, so assertions are made
    against one canonical `$NAME` shape instead of a specific spelling.
    """
    collapsed = re.sub(r"\$+", "$", token)
    return re.sub(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(:?-[^}]*)?\}$", r"$\1\2", collapsed)


def _shell_commands(script: str) -> list[list[str]]:
    """Return each executed command as normalised tokens, redirections removed."""
    commands: list[list[str]] = []
    for line in _split_shell_lines(script):
        for segment in _split_segments(line):
            try:
                tokens = shlex.split(segment, comments=False, posix=True)
            except ValueError:
                continue
            tokens = [_normalise_token(token) for token in tokens]
            tokens = [token for token in tokens if not _REDIRECT_RE.match(token)]
            while tokens and tokens[0] in _SHELL_CONTROL_WORDS:
                tokens.pop(0)
            # A `VAR=value` prefix still executes the rest of the line, so the
            # prefix is only stripped when a real command follows it. A bare
            # assignment is itself an executed statement and is kept whole.
            prefix_count = 0
            while prefix_count < len(tokens) and re.match(
                r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[prefix_count]
            ):
                prefix_count += 1
            if prefix_count < len(tokens):
                tokens = tokens[prefix_count:]
            if tokens:
                commands.append(tokens)
    return commands


def _pull_commands(service: dict) -> list[list[str]]:
    """Executed commands of the model-pull script, in order."""
    return _shell_commands(_pull_script(service))


def _is_show(cmd: list[str]) -> bool:
    """An executed exact lookup of the configured model, not an echoed string."""
    return cmd[:3] == ["ollama", "show", "$TARGET_MODEL"]


def _is_bare_pull(cmd: list[str]) -> bool:
    """A pull that is not wrapped by `timeout`."""
    return cmd[:2] == ["ollama", "pull"]


def _is_grep_presence(cmd: list[str]) -> bool:
    """The old unanchored `ollama list | grep` presence test."""
    return cmd[:2] == ["ollama", "list"] and "grep" in cmd


def _is_bounded_pull(cmd: list[str]) -> bool:
    """``timeout`` + kill escalation + the configured deadline + the configured model.

    Checked on parsed tokens so an ``echo`` of this exact command, a pull of some
    other model, an unwrapped bare ``ollama pull``, a deadline-free ``timeout``, or
    a missing ``--kill-after`` cannot satisfy it. The option may precede or follow
    the duration, which is why the head is searched rather than index-matched.
    """
    if not cmd or cmd[0] != "timeout":
        return False
    try:
        inner = cmd.index("ollama")
    except ValueError:
        return False
    if cmd[inner : inner + 3] != ["ollama", "pull", "$TARGET_MODEL"]:
        return False
    head = cmd[1:inner]
    if not any(re.match(r"^--kill-after=\S+$", token) for token in head):
        return False
    return "$PULL_DEADLINE" in head


def _volume_map(service: dict) -> dict[str, str]:
    """Return ``{container target: host/volume source}`` for one service."""
    mapping: dict[str, str] = {}
    for entry in service.get("volumes") or []:
        if isinstance(entry, str):
            parts = entry.split(":")
            if len(parts) >= 2:
                mapping[parts[1]] = parts[0]
        elif isinstance(entry, dict):
            mapping[str(entry["target"])] = str(entry.get("source", ""))
    return mapping


# ---------------------------------------------------------------------------
# Topology shape
# ---------------------------------------------------------------------------


def test_declares_every_required_service(services: dict) -> None:
    assert REQUIRED_SERVICES <= set(services)


def test_declares_no_unexpected_services(services: dict) -> None:
    # Guards against Milestone 5+ or frontend/lifecycle services sneaking in.
    assert set(services) == REQUIRED_SERVICES


def test_uses_no_obsolete_top_level_version_key(compose: dict) -> None:
    assert "version" not in compose


def test_named_volumes_are_declared(compose: dict) -> None:
    assert {"ollama_data", "prometheus_data", "grafana_data"} <= set(compose["volumes"])


# ---------------------------------------------------------------------------
# techjob-mcp environment + service contract
# ---------------------------------------------------------------------------


def test_mcp_preserves_required_environment_values(services: dict) -> None:
    environment = _env_map(services["techjob-mcp"])
    for key, expected in REQUIRED_MCP_ENV.items():
        assert environment.get(key) == expected, f"techjob-mcp must pin {key}={expected}"


def test_mcp_points_at_the_compose_ollama_service(services: dict) -> None:
    environment = _env_map(services["techjob-mcp"])
    assert environment.get("OLLAMA_URL") == "http://ollama:11434/api/generate"


def test_mcp_builds_from_the_project_dockerfile(services: dict) -> None:
    mcp = services["techjob-mcp"]
    assert mcp["build"]["dockerfile"] == "Dockerfile"
    assert (ROOT / "Dockerfile").is_file()


def test_mcp_keeps_its_application_healthcheck_enabled(services: dict) -> None:
    # The Dockerfile HEALTHCHECK (/health) is inherited by Compose. Disabling it
    # here, or overriding it with a probe that ignores the MCP HTTP surface,
    # would silently break readiness reporting.
    mcp = services["techjob-mcp"]
    assert (mcp.get("healthcheck") or {}).get("disable") is not True
    # Rejoin backslash continuations so the multi-line directive reads as one.
    dockerfile = (ROOT / "Dockerfile").read_text().replace("\\\n", " ")
    directives = [
        line.strip() for line in dockerfile.splitlines() if line.strip().startswith("HEALTHCHECK")
    ]
    assert directives, "Dockerfile must declare a HEALTHCHECK"
    assert any("/health" in line and "curl" in line for line in directives), (
        f"Dockerfile HEALTHCHECK must curl the MCP /health route: {directives}"
    )


def _port_bindings(service: dict) -> list[tuple[str, str]]:
    """Normalise short (``ip:host:ctn``) and long-syntax ports to (host_ip, published)."""
    bindings: list[tuple[str, str]] = []
    for entry in service.get("ports") or []:
        if isinstance(entry, dict):
            bindings.append((str(entry.get("host_ip", "")), str(entry["published"])))
        else:
            parts = str(entry).split(":")
            if len(parts) == 3:
                bindings.append((parts[0], parts[1]))
            elif len(parts) == 2:
                bindings.append(("", parts[0]))
            else:
                bindings.append(("", parts[0]))
    return bindings


def test_observability_ports_stay_loopback_bound(services: dict) -> None:
    """Prometheus/Grafana must not become host-reachable listeners.

    Asserted on normalised bindings, so an equivalent long-syntax port does not
    false-fail the short-syntax form.
    """
    for name in ("prometheus", "grafana"):
        bindings = _port_bindings(services[name])
        assert bindings, f"{name} must publish a port"
        for host_ip, published in bindings:
            assert host_ip == "127.0.0.1", (
                f"{name}:{published} must bind loopback only, got host_ip={host_ip!r}"
            )


# ---------------------------------------------------------------------------
# Ollama healthcheck semantics (the Milestone 4 defect class)
# ---------------------------------------------------------------------------


def test_ollama_declares_a_healthcheck(services: dict) -> None:
    assert "healthcheck" in services["ollama"]


def test_ollama_healthcheck_is_bounded(services: dict) -> None:
    healthcheck = services["ollama"]["healthcheck"]
    for key in ("test", "interval", "timeout", "retries"):
        assert key in healthcheck, f"ollama healthcheck must declare {key}"
    assert int(healthcheck["retries"]) > 0


def test_ollama_healthcheck_probes_daemon_readiness_only(services: dict) -> None:
    """Readiness must be satisfiable with an empty model store.

    Root cause under test: gating ``ollama`` health on ``ollama list | grep <model>``
    makes the daemon permanently unhealthy on any volume that does not already
    hold the probed name. Because ``ollama-model-pull`` needs a *healthy* ollama
    to run, that turns into a startup deadlock instead of a readiness signal.

    Asserted structurally (no filtering, no variable expansion) rather than by
    enumerating model names, so a legitimate future probe is not blocked and a
    renamed model gate cannot slip through.
    """
    probe = _healthcheck_text(services["ollama"])
    assert "ollama list" in probe, "ollama healthcheck must probe the daemon API"
    assert "grep" not in probe, f"ollama healthcheck must not filter output: {probe}"
    assert "${" not in probe and "$$" not in probe, (
        f"ollama healthcheck must not expand a model variable: {probe}"
    )
    for token in MODEL_TOKENS:
        assert token not in probe, f"ollama healthcheck must not gate on {token!r}"


# ---------------------------------------------------------------------------
# ollama-model-pull dependency semantics
# ---------------------------------------------------------------------------


def test_model_pull_waits_for_ollama_health_not_merely_start(services: dict) -> None:
    conditions = _depends_on(services["ollama-model-pull"])
    assert conditions.get("ollama") == "service_healthy"


def test_model_pull_delegates_readiness_to_compose(services: dict) -> None:
    """No in-script polling: ``service_healthy`` is the bounded gate."""
    commands = _pull_commands(services["ollama-model-pull"])
    assert not [c for c in commands if "sleep" in c], (
        "ollama-model-pull must not use sleep-based readiness"
    )


def test_model_pull_detects_the_model_exactly(services: dict) -> None:
    """Presence must be an executed exact lookup, not a substring or regex match.

    Root cause under test: ``ollama list | grep -q "$MODEL"`` is an unanchored
    regular expression. With ``qwen2.5-coder:3b`` installed, grepping the partial
    name ``qwen2.5`` matched, so the helper exited 0 and released techjob-mcp
    through ``service_completed_successfully`` without the configured model ever
    being pulled.

    Asserted over parsed shell commands: an ``echo`` containing this very text, or
    a lookup of some other model, is not an executed ``ollama show``.
    """
    commands = _pull_commands(services["ollama-model-pull"])
    assert not [c for c in commands if _is_grep_presence(c)], (
        "ollama-model-pull must not test presence with the unanchored `ollama list | grep` form"
    )
    shows = [index for index, cmd in enumerate(commands) if _is_show(cmd)]
    pulls = [index for index, cmd in enumerate(commands) if _is_bounded_pull(cmd)]
    assert shows, "ollama-model-pull must test the configured model with an executed `ollama show`"
    assert pulls, "ollama-model-pull must perform the bounded pull"
    assert shows[0] < pulls[0], "ollama-model-pull must check the configured model before pulling"
    assert shows[-1] > pulls[-1], (
        "ollama-model-pull must re-verify the configured model after pulling, so a truncated "
        "or failed transfer cannot be reported as readiness"
    )


def test_model_pull_deadline_is_bounded_and_configurable(services: dict) -> None:
    """A stalled registry must not block startup forever.

    ``techjob-mcp`` waits on ``service_completed_successfully``, so an unbounded
    ``ollama pull`` would hang the whole stack. The pull must be wrapped by
    ``timeout`` with kill escalation and the configured deadline, and a clean
    checkout must be able to override that deadline.
    """
    commands = _pull_commands(services["ollama-model-pull"])
    bounded = [cmd for cmd in commands if _is_bounded_pull(cmd)]
    assert bounded, (
        "ollama pull must run as `timeout --kill-after=<dur> $PULL_DEADLINE ollama pull "
        "$TARGET_MODEL`, not bare or with a constant deadline"
    )
    # No pull may escape the wrapper.
    assert not [cmd for cmd in commands if _is_bare_pull(cmd)], (
        "every `ollama pull` must be wrapped by `timeout`"
    )
    environment = _env_map(services["ollama-model-pull"])
    deadline = environment.get("OLLAMA_PULL_TIMEOUT_SECONDS", "")
    assert "OLLAMA_PULL_TIMEOUT_SECONDS" in deadline
    assert ":-" in deadline, "the pull deadline needs a default for a clean clone"


def test_model_pull_rejects_a_deadline_that_would_disable_the_bound(services: dict) -> None:
    """Every mis-set deadline must still yield a real bound.

    GNU ``timeout 0`` means "no timeout at all", so a permissive reading of
    ``OLLAMA_PULL_TIMEOUT_SECONDS`` silently restores an indefinitely blocking
    pull and contradicts the bounded-startup requirement. Overlong all-digit
    values additionally overflow shell ``test`` comparisons and slip past a
    numeric guard alone. Each case must be normalised before the pull runs.
    """
    commands = _pull_commands(services["ollama-model-pull"])
    pull_index = next(
        (i for i, cmd in enumerate(commands) if _is_bounded_pull(cmd)),
        -1,
    )
    assert pull_index != -1, "the bounded pull command must exist"
    head = commands[:pull_index]

    def _test(cmd: list[str], operator: str) -> bool:
        return cmd[0] == "[" and operator in cmd

    required = {
        "non-numeric deadline": lambda cmd: "*[!0-9]*)" in cmd,
        # The overlong guard must inspect the shell length expansion `${#VAR}`.
        "overlong deadline": lambda cmd: _test(cmd, "-ge")
        and any("#PULL_DEADLINE" in token for token in cmd),
        "zero deadline": lambda cmd: _test(cmd, "-le"),
        "deadline ceiling": lambda cmd: _test(cmd, "-gt"),
    }
    for label, predicate in required.items():
        assert any(predicate(cmd) for cmd in head), (
            f"ollama-model-pull must guard against a {label} before pulling"
        )

    # The zero and ceiling comparisons must act on the deadline variable itself,
    # with a positive numeric threshold as the operator's right-hand operand.
    for operator in ("-le", "-gt"):
        guarded = [cmd for cmd in head if _test(cmd, operator)]
        assert guarded, f"a {operator} deadline comparison must exist"
        for cmd in guarded:
            assert "$PULL_DEADLINE" in cmd, f"the {operator} guard must compare $PULL_DEADLINE"
            operand = cmd[cmd.index(operator) + 1 :]
            threshold = next((token for token in operand if token.isdigit()), "")
            assert threshold.isdigit() and int(threshold) >= 0, (
                f"the {operator} guard needs a numeric threshold"
            )
        if operator == "-gt":
            ceilings = [
                int(next(token for token in cmd[cmd.index("-gt") + 1 :] if token.isdigit()))
                for cmd in guarded
            ]
            assert all(value > 0 for value in ceilings), "the deadline ceiling must be positive"


def test_model_pull_fails_fast_when_the_pull_fails(services: dict) -> None:
    """A failed pull must not report success.

    Without ``set -e`` the helper exits 0 even when ``ollama pull`` errors, so a
    completion-gated consumer would start against a missing model. Asserted on
    parsed commands, so shell-equivalent whitespace cannot break the check.
    """
    commands = _pull_commands(services["ollama-model-pull"])
    assert ["set", "-e"] in commands, "ollama-model-pull must run with set -e"
    assert any(_is_bounded_pull(cmd) for cmd in commands), (
        "ollama-model-pull must perform the bounded pull"
    )


def test_model_pull_targets_the_configured_model_with_a_default(services: dict) -> None:
    """The pull target must follow the configured model, with a clean-clone default."""
    environment = _env_map(services["ollama-model-pull"])
    assert environment.get("OLLAMA_HOST") == "http://ollama:11434"
    declared_model = environment.get("OLLAMA_MODEL", "")
    assert "OLLAMA_MODEL" in declared_model, "pull target must follow the configured model"
    assert ":-" in declared_model, "pull target needs a default for a clean clone without .env"
    commands = _pull_commands(services["ollama-model-pull"])
    assert any(
        cmd[0].startswith("TARGET_MODEL=") and "OLLAMA_MODEL" in cmd[0] and ":-" in cmd[0]
        for cmd in commands
    ), "ollama-model-pull must resolve TARGET_MODEL from OLLAMA_MODEL with a default"


def test_model_pull_shares_the_ollama_model_volume(services: dict) -> None:
    assert "ollama_data" in _volume_map(services["ollama-model-pull"]).values()


def test_model_pull_does_not_restart_forever(services: dict) -> None:
    # A one-shot helper must terminate; `service_completed_successfully` only
    # has meaning for a container that actually exits.
    assert str(services["ollama-model-pull"]["restart"]) == "no"


# ---------------------------------------------------------------------------
# techjob-mcp must wait for model preparation, not just a live daemon
# ---------------------------------------------------------------------------


def test_mcp_waits_for_model_preparation_to_complete(services: dict) -> None:
    conditions = _depends_on(services["techjob-mcp"])
    assert conditions.get("ollama-model-pull") == "service_completed_successfully"


def test_mcp_still_waits_for_ollama_health(services: dict) -> None:
    conditions = _depends_on(services["techjob-mcp"])
    assert conditions.get("ollama") == "service_healthy"


# ---------------------------------------------------------------------------
# Prometheus / Grafana wiring
# ---------------------------------------------------------------------------


def test_prometheus_scrapes_the_mcp_metrics_endpoint(services: dict) -> None:
    mounts = _volume_map(services["prometheus"])
    assert "/etc/prometheus/prometheus.yml" in mounts
    prometheus_config = (ROOT / "deploy/prometheus/prometheus.yml").read_text()
    assert "techjob-mcp:8000" in prometheus_config
    assert "metrics_path: /metrics" in prometheus_config


def test_prometheus_config_mount_source_exists(services: dict) -> None:
    source = _volume_map(services["prometheus"])["/etc/prometheus/prometheus.yml"]
    assert (ROOT / source.lstrip("./")).is_file(), f"missing Prometheus mount: {source}"


def test_grafana_provisioning_mounts_exist(services: dict) -> None:
    mounts = _volume_map(services["grafana"])
    for target in ("/etc/grafana/provisioning", "/var/lib/grafana/dashboards"):
        source = mounts[target]
        assert (ROOT / source.lstrip("./")).is_dir(), f"missing Grafana mount: {source}"


def test_observability_dependency_chain_is_sane(services: dict) -> None:
    assert "techjob-mcp" in _depends_on(services["prometheus"])
    assert "prometheus" in _depends_on(services["grafana"])


# ---------------------------------------------------------------------------
# Executable Compose validation (skips honestly when Docker is absent)
# ---------------------------------------------------------------------------


def _scratch_root() -> Path:
    """Root for Docker-visible scratch projects.

    Some Docker installs (notably the snap sandbox) cannot read the system
    ``/tmp``, and pytest's default ``tmp_path`` would then produce a false
    "file not found" instead of a real verdict. ``TMPDIR`` is respected first so
    CI can point this at a Docker-readable location.
    """
    root = Path(os.environ.get("TMPDIR") or (Path.home() / ".cache" / "techjobmcp-tests"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _synthetic_project(*, with_env: bool) -> Path:
    """Copy the real compose file into an isolated, uniquely named project dir.

    Exercises Compose interpolation, healthcheck and depends_on schema without
    depending on the developer-local ``.env`` secrets. ``mkdtemp`` keeps the
    directory unique so concurrent pytest-xdist workers cannot delete each
    other's active project.
    """
    project = Path(tempfile.mkdtemp(prefix="compose-topology-", dir=str(_scratch_root())))
    shutil.copy(COMPOSE_FILE, project / "docker-compose.yml")
    if with_env:
        (project / ".env").write_text("OLLAMA_MODEL=synthetic-model:1\n")
    return project


def _remove_project(project: Path) -> None:
    shutil.rmtree(project, ignore_errors=True)


def _run_compose(project: Path, *args: str) -> subprocess.CompletedProcess:
    """Render the compose file without raising; callers assert on returncode.

    Compose gives the process environment priority over the project ``.env``, so
    ambient values would override the synthetic ones. Other tests in this repo
    load the repository ``.env`` into ``os.environ`` via ``load_dotenv()``, so
    every variable this compose file interpolates is scrubbed from the child
    environment to keep interpolation deterministic.
    """
    environment = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_")}
    return subprocess.run(
        ["docker", "compose", "-f", str(project / "docker-compose.yml"), *args],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=environment,
    )


def _compose_cli_available() -> bool:
    """True only when Docker *and* the Compose plugin both work.

    Presence of the ``docker`` binary alone does not prove ``docker compose``
    exists, so a plugin-less host would fail these tests instead of skipping.
    """
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # A hung or missing CLI is an unavailable toolchain, not a topology defect.
        return False
    return probe.returncode == 0


requires_compose = pytest.mark.skipif(
    not _compose_cli_available(), reason="docker compose CLI unavailable"
)


@requires_compose
def test_compose_config_validates_quietly() -> None:
    project = _synthetic_project(with_env=True)
    try:
        result = _run_compose(project, "config", "--quiet")
        assert result.returncode == 0, result.stderr
    finally:
        _remove_project(project)


@requires_compose
def test_compose_config_validates_without_a_local_env_file() -> None:
    """A clean clone has no ``.env``; the M4 gate is ``docker compose config``.

    If the env file is mandatory, configuration validation fails before any
    topology can be checked, so the roadmap gate is unrunnable from a fresh
    checkout. The required MCP values live in the compose file itself.
    """
    project = _synthetic_project(with_env=False)
    try:
        result = _run_compose(project, "config", "--quiet")
        assert result.returncode == 0, result.stderr
    finally:
        _remove_project(project)


@requires_compose
def test_rendered_topology_keeps_health_and_dependency_semantics() -> None:
    """Assert the delivered (interpolated) form, not just the raw YAML.

    Compose un-escapes ``$$`` to ``$`` before Docker executes a healthcheck, and
    resolves ``${VAR:-default}`` from the project env file, so the rendered
    document is what the containers actually run.
    """
    project = _synthetic_project(with_env=True)
    try:
        result = _run_compose(project, "config", "--format", "json")
        assert result.returncode == 0, result.stderr
    finally:
        _remove_project(project)
    rendered = json.loads(result.stdout)["services"]

    probe = " ".join(str(part) for part in rendered["ollama"]["healthcheck"]["test"])
    assert "$$" not in probe, f"rendered probe must be shell-executable, got: {probe}"
    for token in MODEL_TOKENS:
        assert token not in probe, f"rendered probe must stay model-free, got: {probe}"

    assert _env_map(rendered["ollama-model-pull"])["OLLAMA_MODEL"] == "synthetic-model:1"
    assert _env_map(rendered["techjob-mcp"])["OLLAMA_MODEL"] == "synthetic-model:1"
    assert _env_map(rendered["techjob-mcp"])["SOURCE_TIMEOUT_SECONDS"] == "15.0"
    assert _depends_on(rendered["ollama-model-pull"])["ollama"] == "service_healthy"
    assert _depends_on(rendered["techjob-mcp"])["ollama-model-pull"] == (
        "service_completed_successfully"
    )
