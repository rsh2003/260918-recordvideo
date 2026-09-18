#!/usr/bin/env python3
"""6단계(선택): Gradio 기반 웹 UI로 스무고개 플레이.

step4_play.py와 게임 로직은 완전히 동일하다. 다른 점은 콘솔 input() 대신
브라우저에서 버튼을 눌러 답하고, 진행 기록이 텍스트박스에 실시간으로 표시된다는 것뿐이다.

설치:  pip install gradio
실행:  python src/app_ui.py
환경:  ADAPTER(기본 output/adapter)  MAXTURNS(기본 20)  SERVER_PORT(기본 7860)

ADAPTER는 LoRA 어댑터 폴더 경로를 가리키면 된다(이미 Gemma 4 E4B 베이스가 설치된
환경이라면 unsloth가 베이스 모델 + 어댑터를 함께 불러온다. step4_play.py와 동일한 방식).
"""
import json
import os
import re

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")

import gradio as gr                              # noqa: E402
import unsloth                                   # noqa: E402
from unsloth import FastModel                    # noqa: E402
import torch                                     # noqa: E402

from common import SYS_QUESTIONER, user_turn, OUTPUT   # noqa: E402

ADAPTER = os.environ.get("ADAPTER", str(OUTPUT / "adapter"))
MAXTURNS = int(os.environ.get("MAXTURNS", "20"))
JSON_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')

print(f"[로딩] {ADAPTER}", flush=True)
model, tok = FastModel.from_pretrained(model_name=ADAPTER, max_seq_length=1024,
                                       load_in_4bit=True, device_map={"": 0})
model.eval()
inner = getattr(tok, "tokenizer", tok)

ASK_OPTS = ["네", "아니오", "가끔", "모름"]
FLAG_OPTS = ["네, 맞습니다", "아니오, 틀렸습니다"]


def next_action(hist):
    """step4_play.py와 동일: 기록을 넣어 모델의 다음 행동(JSON) 하나를 받는다."""
    msgs = [{"role": "system", "content": SYS_QUESTIONER},
            {"role": "user", "content": user_turn(hist)}]
    text = inner.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = inner(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=160, do_sample=False,
                             pad_token_id=inner.eos_token_id)
    raw = inner.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = JSON_RE.search(raw.replace("\n", " "))
    if not m:
        return None, raw[:120]
    try:
        return json.loads(m.group(0)), None
    except Exception:
        return None, raw[:120]


def render_log(log):
    return "\n".join(log) if log else "'새 게임 시작'을 눌러 시작하세요."


def new_state():
    return {"hist": "", "turn": 0, "log": [], "pending": None, "done": False}


def advance(state):
    """모델이 형식 오류 없이 질문/추측을 내놓을 때까지 진행하고, 사람의 답을 기다리는 지점에서 멈춘다."""
    if state["done"]:
        return render_log(state["log"]), gr.update(visible=False), gr.update(visible=False), state

    if state["turn"] >= MAXTURNS:
        state["log"].append("── 턴 초과: 맞히지 못했습니다 ──")
        state["done"] = True
        return render_log(state["log"]), gr.update(visible=False), gr.update(visible=False), state

    act, raw = next_action(state["hist"])
    state["turn"] += 1

    if act is None:
        state["hist"] += "(형식오류)\n"
        state["log"].append(f"[{state['turn']}] (형식 오류: {raw})")
        return advance(state)  # 이번 턴은 소모하고 바로 다음 시도로 넘어간다

    if act.get("action") == "flag":
        g = str(act.get("guess", "")).strip()
        state["pending"] = {"type": "flag", "guess": g}
        state["log"].append(f"[{state['turn']}] 🤖 모델 추측: '{g}' — 맞나요?")
        return (render_log(state["log"]),
                gr.update(visible=True, choices=FLAG_OPTS, label="정답 확인", value=None),
                gr.update(visible=True), state)

    q = str(act.get("question", "")).strip()
    state["pending"] = {"type": "ask", "question": q}
    state["log"].append(f"[{state['turn']}] 🤖 모델 질문: {q}")
    return (render_log(state["log"]),
            gr.update(visible=True, choices=ASK_OPTS, label="답변 선택", value=None),
            gr.update(visible=True), state)


def start_game():
    return advance(new_state())


def submit_answer(choice, state):
    if not choice or state.get("pending") is None or state.get("done"):
        return render_log(state["log"]), gr.update(), gr.update(), state

    p = state["pending"]
    if p["type"] == "flag":
        if choice.startswith("네"):
            state["log"].append(f"→ 정답! {state['turn']}턴 만에 맞혔습니다 🎉")
            state["done"] = True
            return render_log(state["log"]), gr.update(visible=False), gr.update(visible=False), state
        state["hist"] += f"추측: {p['guess']} -> 아닙니다\n"
    else:
        state["hist"] += f"Q: {p['question']} -> {choice}\n"

    state["pending"] = None
    return advance(state)


with gr.Blocks(title="스무고개 sLLM") as demo:
    gr.Markdown(
        "## 🤔 스무고개 sLLM\n"
        "마음속으로 단어 하나를 정하세요(학습 단어 목록: `data/words_917.txt`). "
        "모델이 질문하면 아래 버튼으로 답하세요."
    )
    game_state = gr.State(new_state())
    log_box = gr.Textbox(label="진행 기록", lines=16, interactive=False)
    answer_choice = gr.Radio(choices=[], label="답변 선택", visible=False)
    submit_btn = gr.Button("답변 제출", variant="primary", visible=False)
    restart_btn = gr.Button("🔄 새 게임 시작")

    restart_btn.click(start_game, outputs=[log_box, answer_choice, submit_btn, game_state])
    submit_btn.click(submit_answer, inputs=[answer_choice, game_state],
                     outputs=[log_box, answer_choice, submit_btn, game_state])
    demo.load(start_game, outputs=[log_box, answer_choice, submit_btn, game_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("SERVER_PORT", "7860")))
