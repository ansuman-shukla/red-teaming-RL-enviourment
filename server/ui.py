"""Custom Gradio UI for RedTeamEnv."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

try:
    from ..models import RedTeamAction
    from .task_definitions import TASK_SPECS, task_names
except ImportError:  # pragma: no cover
    from models import RedTeamAction
    from server.task_definitions import TASK_SPECS, task_names


CUSTOM_CSS = """
:root {
  --rt-bg: linear-gradient(145deg, #fbf5eb 0%, #f3e8d8 52%, #e6d6bf 100%);
  --rt-panel: rgba(255, 251, 245, 0.94);
  --rt-panel-strong: rgba(255, 248, 239, 0.985);
  --rt-border: rgba(120, 77, 34, 0.14);
  --rt-ink: #2a1b12;
  --rt-muted: #6d584a;
  --rt-accent: #b14a1f;
  --rt-accent-deep: #7f2d11;
  --rt-ok: #2f6a46;
  --rt-shadow: 0 22px 54px rgba(66, 34, 8, 0.1);
  --rt-input: #fffaf3;
  --rt-input-border: rgba(120, 77, 34, 0.16);
  --rt-surface: rgba(255, 255, 255, 0.58);
}

html,
body,
.gradio-container,
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .main {
  background: var(--rt-bg);
  color: var(--rt-ink) !important;
}

body {
  color-scheme: light;
}

.gradio-container {
  width: min(1440px, calc(100vw - 40px)) !important;
  max-width: min(1440px, calc(100vw - 40px)) !important;
  margin: 0 auto !important;
  padding: 28px 0 40px 0 !important;
}

.gradio-container > div,
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .main {
  max-width: none !important;
  width: 100% !important;
}

.rt-shell {
  color: var(--rt-ink);
}

.rt-hero,
.rt-panel,
.rt-metric {
  background: var(--rt-panel);
  border: 1px solid var(--rt-border);
  border-radius: 24px;
  box-shadow: var(--rt-shadow);
  backdrop-filter: blur(12px);
}

.rt-hero {
  padding: 28px 30px 18px 30px;
  margin-bottom: 18px;
  background:
    radial-gradient(circle at top right, rgba(177, 74, 31, 0.16), transparent 38%),
    radial-gradient(circle at left bottom, rgba(87, 120, 73, 0.12), transparent 35%),
    var(--rt-panel-strong);
}

.rt-kicker {
  margin: 0 0 8px 0;
  color: var(--rt-accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.rt-hero h1,
.rt-hero h2,
.rt-panel h3,
.rt-panel h4 {
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  letter-spacing: 0.02em;
  color: var(--rt-ink);
}

.rt-hero h1 {
  margin: 0 0 8px 0;
  font-size: 36px;
  line-height: 1.05;
}

.rt-hero p,
.rt-panel p,
.rt-panel li,
.rt-metric {
  color: var(--rt-muted);
  font-size: 15px;
}

.rt-grid {
  gap: 18px;
  align-items: stretch;
}

.rt-panel {
  padding: 18px 18px 12px 18px;
  height: 100%;
}

.rt-panel h3 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 22px;
}

.rt-subnote {
  margin: 0 0 14px 0;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(177, 74, 31, 0.06);
  color: var(--rt-muted);
  font-size: 14px;
}

.rt-metrics {
  gap: 12px;
}

.rt-metric {
  padding: 14px 16px;
  min-height: 94px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(255, 247, 238, 0.88));
}

.rt-metric h4 {
  margin: 0 0 8px 0;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--rt-muted);
}

.rt-metric p {
  margin: 0;
  color: var(--rt-ink);
  font-size: 26px;
  font-weight: 700;
}

.rt-banner {
  background: rgba(177, 74, 31, 0.08);
  border: 1px solid rgba(177, 74, 31, 0.14);
  border-radius: 16px;
  padding: 12px 14px;
  margin-top: 14px;
}

.rt-banner strong {
  color: var(--rt-accent-deep);
}

.rt-history {
  background: var(--rt-surface);
  border: 1px solid var(--rt-border);
  border-radius: 18px;
  padding: 14px 16px;
}

.rt-history code {
  white-space: pre-wrap;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container label,
.gradio-container legend,
.gradio-container .form label,
.gradio-container .form legend {
  color: var(--rt-ink) !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .border,
.gradio-container .gr-box,
.gradio-container .gr-panel {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

.gradio-container .primary {
  background: linear-gradient(135deg, var(--rt-accent), var(--rt-accent-deep)) !important;
  border: none !important;
  color: #fff8f2 !important;
  box-shadow: 0 14px 26px rgba(127, 45, 17, 0.22) !important;
}

.gradio-container .secondary {
  background: rgba(255, 250, 243, 0.88) !important;
  border-color: rgba(122, 39, 16, 0.22) !important;
  color: var(--rt-accent-deep) !important;
}

.gradio-container button {
  border-radius: 14px !important;
  min-height: 44px !important;
  font-weight: 700 !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap {
  border-radius: 18px !important;
}

.gradio-container .form {
  gap: 14px !important;
}

.gradio-container .form > div,
.gradio-container .input-container,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  width: 100% !important;
  box-sizing: border-box !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container .input-container,
.gradio-container .wrap-inner {
  background: var(--rt-input) !important;
  border: 1px solid var(--rt-input-border) !important;
  color: var(--rt-ink) !important;
  border-radius: 16px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: rgba(109, 88, 74, 0.74) !important;
}

.gradio-container [data-testid="dropdown"],
.gradio-container [data-testid="textbox"] {
  margin-bottom: 4px !important;
}

.gradio-container [data-testid="textbox"] textarea {
  min-height: 148px !important;
}

.gradio-container .accordion,
.gradio-container .label-wrap,
.gradio-container .tabs {
  border-color: var(--rt-border) !important;
}

.gradio-container .label-wrap > label {
  font-weight: 700 !important;
}

.gradio-container .generating,
.gradio-container .pending {
  background: rgba(177, 74, 31, 0.08) !important;
}

footer {
  display: none !important;
}

@media (max-width: 900px) {
  .gradio-container {
    width: min(100vw - 20px, 100%) !important;
    max-width: min(100vw - 20px, 100%) !important;
    padding: 16px 0 28px 0 !important;
  }

  .rt-hero {
    padding: 20px 18px 16px 18px;
  }

  .rt-hero h1 {
    font-size: 28px;
  }
}
"""


def build_redteam_gradio_app(
    web_manager: Any,
    action_fields: Any,
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the full RedTeamEnv web UI mounted at /web."""
    del action_fields, is_chat_env, quick_start_md

    async def reset_env(
        task_name: str,
        seed_value: str,
    ) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
        try:
            reset_kwargs: dict[str, Any] = {"task_name": task_name}
            normalized_seed = seed_value.strip()
            if normalized_seed:
                reset_kwargs["seed"] = int(normalized_seed)
            payload = await web_manager.reset_environment(reset_kwargs)
            observation = _merge_step_fields(payload)
            state = web_manager.get_state()
            task = TASK_SPECS[task_name]
            seed_suffix = ""
            if isinstance(state, dict) and state.get("seed") is not None:
                seed_suffix = f" Seed: `{state['seed']}`."
            return _render_outputs(
                observation=observation,
                state=state,
                status=f"Episode reset for `{task_name}`.{seed_suffix}",
                task_description=(
                    f"Max turns: {task.max_turns} | Preferred strategies: "
                    f"{', '.join(task.preferred_strategies)}"
                ),
            )
        except Exception as exc:
            return _render_error(
                f"Reset failed for `{task_name}`: {exc}",
                task_description=_task_description(task_name),
            )

    async def step_env(
        task_name: str,
        strategy: str,
        prompt: str,
        turn_context: str,
    ) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
        if not prompt.strip():
            return _render_error(
                "Prompt is required before stepping the environment.",
                task_description=_task_description(task_name),
            )

        try:
            payload = await web_manager.step_environment(
                RedTeamAction(
                    strategy=strategy,
                    prompt=prompt,
                    turn_context=turn_context,
                ).model_dump(mode="json")
            )
            observation = _merge_step_fields(payload)
            state = web_manager.get_state()
            status = _summarize_observation(observation)
            return _render_outputs(
                observation=observation,
                state=state,
                status=status,
                task_description=_task_description(task_name),
            )
        except Exception as exc:
            return _render_error(
                f"Step failed: {exc}",
                task_description=_task_description(task_name),
            )

    with gr.Blocks(
        title=title,
        fill_width=True,
    ) as blocks:
        with gr.Column(elem_classes="rt-shell"):
            gr.HTML(
                f"""
<div class="rt-hero">
  <div class="rt-kicker">Red Team Evaluation Console</div>
  <h1>{title}</h1>
  <p>{metadata.description}</p>
  <div class="rt-banner">
    <strong>Manual validation flow:</strong> reset a task, send one strategy-bound prompt,
    inspect the target response, then confirm reward, flags, and remaining attempts.
  </div>
</div>
                """,
            )

            with gr.Row(elem_classes="rt-grid"):
                with gr.Column(scale=5, elem_classes="rt-panel"):
                    gr.Markdown("### Control Panel")
                    gr.Markdown(
                        '<div class="rt-subnote">Choose a task, pick the attack framing, '
                        "then iterate on the prompt while watching reward, flags, and attempts.</div>"
                    )
                    with gr.Row():
                        task_name = gr.Dropdown(
                            choices=list(task_names()),
                            value="stereotype_probe",
                            label="Task",
                        )
                        strategy = gr.Dropdown(
                            choices=[
                                "roleplay",
                                "obfuscation",
                                "decomposition",
                                "narrative_wrap",
                                "multilingual",
                                "direct",
                            ],
                            value="roleplay",
                            label="Strategy",
                        )
                        reset_seed = gr.Textbox(
                            label="Seed",
                            placeholder="Optional, e.g. 1616021387",
                        )
                    prompt = gr.Textbox(
                        label="Adversarial Prompt",
                        lines=7,
                        placeholder="Write the next probe you want the target model to answer.",
                    )
                    turn_context = gr.Textbox(
                        label="Turn Context",
                        lines=3,
                        placeholder="Optional grooming or role-play context carried into this turn.",
                    )
                    with gr.Row():
                        reset_button = gr.Button("Reset Episode", variant="secondary")
                        step_button = gr.Button("Run Step", variant="primary")

                with gr.Column(scale=7, elem_classes="rt-panel"):
                    gr.Markdown("### Live Response")
                    status = gr.Markdown(
                        "Reset an episode to start. The target response and validation values will appear here."
                    )
                    task_description = gr.Markdown(_task_description("stereotype_probe"))
                    target_response = gr.Textbox(
                        label="Target Response",
                        lines=10,
                        interactive=False,
                        placeholder="The assistant response for the last step will appear here.",
                    )
                    message = gr.Textbox(
                        label="Environment Message",
                        lines=2,
                        interactive=False,
                    )

            with gr.Row(elem_classes="rt-metrics"):
                reward = gr.HTML(
                    _metric_html("Reward", "0.00"),
                    elem_classes="rt-metric",
                )
                attack_score = gr.HTML(
                    _metric_html("Attack Score", "0.00"),
                    elem_classes="rt-metric",
                )
                attempts_remaining = gr.HTML(
                    _metric_html("Attempts Left", "--"),
                    elem_classes="rt-metric",
                )
                done = gr.HTML(
                    _metric_html("Done", "false"),
                    elem_classes="rt-metric",
                )

            with gr.Row(elem_classes="rt-grid"):
                with gr.Column(scale=6, elem_classes="rt-panel"):
                    gr.Markdown("### Flags")
                    compliance_flags = gr.Textbox(
                        label="Verified Flags",
                        lines=3,
                        interactive=False,
                    )
                    gr.Markdown("### Conversation Trace")
                    conversation_history = gr.Markdown(
                        value=_format_history([]),
                        elem_classes="rt-history",
                    )

                with gr.Column(scale=6, elem_classes="rt-panel"):
                    with gr.Accordion("Observation JSON", open=False):
                        observation_json = gr.Code(
                            value="{}",
                            label="Observation",
                            language="json",
                            interactive=False,
                        )
                    with gr.Accordion("State JSON", open=False):
                        state_json = gr.Code(
                            value="{}",
                            label="State",
                            language="json",
                            interactive=False,
                        )

            reset_button.click(
                reset_env,
                inputs=[task_name, reset_seed],
                outputs=[
                    status,
                    task_description,
                    target_response,
                    message,
                    reward,
                    attack_score,
                    attempts_remaining,
                    done,
                    compliance_flags,
                    conversation_history,
                    observation_json,
                    state_json,
                ],
            )
            step_button.click(
                step_env,
                inputs=[task_name, strategy, prompt, turn_context],
                outputs=[
                    status,
                    task_description,
                    target_response,
                    message,
                    reward,
                    attack_score,
                    attempts_remaining,
                    done,
                    compliance_flags,
                    conversation_history,
                    observation_json,
                    state_json,
                ],
            )

    return blocks


def _render_outputs(
    *,
    observation: dict[str, Any],
    state: dict[str, Any],
    status: str,
    task_description: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    flags = observation.get("compliance_flags") or []
    return (
        status,
        task_description,
        observation.get("target_response", ""),
        observation.get("message", ""),
        _metric_html("Reward", f"{float(observation.get('reward', 0.0)):.2f}"),
        _metric_html(
            "Attack Score",
            f"{float(observation.get('attack_score', 0.0)):.2f}",
        ),
        _metric_html(
            "Attempts Left",
            str(observation.get("attempts_remaining", "--")),
        ),
        _metric_html("Done", str(bool(observation.get("done", False))).lower()),
        ", ".join(flags) if flags else "No verified flags on the last turn.",
        _format_history(observation.get("conversation_history", [])),
        json.dumps(observation, indent=2),
        json.dumps(state, indent=2),
    )


def _merge_step_fields(payload: dict[str, Any]) -> dict[str, Any]:
    observation = dict(payload.get("observation", {}))
    for field in ("reward", "done", "metadata"):
        if field in payload and field not in observation:
            observation[field] = payload[field]
    return observation


def _render_error(
    status: str,
    *,
    task_description: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    return (
        status,
        task_description,
        "",
        status,
        _metric_html("Reward", "0.00"),
        _metric_html("Attack Score", "0.00"),
        _metric_html("Attempts Left", "--"),
        _metric_html("Done", "false"),
        "No verified flags on the last turn.",
        _format_history([]),
        "{}",
        "{}",
    )


def _metric_html(label: str, value: str) -> str:
    return f"<h4>{label}</h4><p>{value}</p>"


def _task_description(task_name: str) -> str:
    task = TASK_SPECS[task_name]
    return (
        f"**Task:** `{task.name}`  \n"
        f"**Max turns:** `{task.max_turns}`  \n"
        f"**Preferred strategies:** `{', '.join(task.preferred_strategies)}`"
    )


def _format_history(conversation_history: list[dict[str, Any]]) -> str:
    if not conversation_history:
        return "_No conversation history yet._"

    lines: list[str] = []
    for index, entry in enumerate(conversation_history[-8:], start=1):
        role = str(entry.get("role", "unknown")).upper()
        content = str(entry.get("content", "")).strip() or "(empty)"
        lines.append(f"**{index}. {role}**\n\n{content}")
    return "\n\n---\n\n".join(lines)


def _summarize_observation(observation: dict[str, Any]) -> str:
    flags = observation.get("compliance_flags", [])
    reward = float(observation.get("reward", 0.0))
    score = float(observation.get("attack_score", 0.0))
    attempts = observation.get("attempts_remaining", 0)
    done = str(bool(observation.get("done", False))).lower()
    return (
        f"Turn recorded. reward={reward:.2f} attack_score={score:.2f} "
        f"attempts_remaining={attempts} done={done} flags={flags or []}"
    )
