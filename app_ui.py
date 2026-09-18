#!/usr/bin/env python3
"""스무고개 웹 UI (Gradio). scripts/play.py와 로직은 동일, 콘솔 input() 대신 브라우저 버튼으로 답한다.

이 파일은 20q_llm 폴더를 압축 풀었을 때 생기는 scripts/ 폴더 안에 넣고 실행한다
(scripts/play.py, scripts/eval.py와 같은 위치).

설치:  pip install gradio
실행:  python scripts/app_ui.py
옵션:  ADAPTER=<경로>  다른 어댑터로 실행 (기본: 20q_llm/adapter/)
       MAXTURNS=20  SERVER_PORT=7860
"""
import json
import os
import re
from pathlib import Path

try:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]   # 20q_llm/
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")

import gradio as gr                              # noqa: E402
import unsloth                                   # noqa: E402  (반드시 torch보다 먼저)
from unsloth import FastModel                    # noqa: E402
import torch                                     # noqa: E402

ADAPTER = os.environ.get("ADAPTER", str(ROOT / "adapter"))
MAXTURNS = int(os.environ.get("MAXTURNS", "20"))
JSON_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')

SYS = ("당신은 한국어 스무고개 '질문자'입니다. 정답은 일상에서 흔히 보는 '구체적인 명사' 하나입니다.\n"
       "지금까지의 질문/답 기록을 보고, 남은 가능성을 가장 잘 둘로 가르는 예/아니오 질문을 하세요.\n"
       "후보가 충분히 좁혀졌다고 판단되면 더 질문하지 말고 즉시 flag로 정답을 추측하세요. "
       "확신이 100%가 아니어도, 좁혀졌다면 가장 가능성 높은 단어를 적극적으로 추측하는 편이 좋습니다. "
       "질문만 반복하지 말고 충분히 좁혀졌으면 반드시 추측하세요. 매 턴 JSON 한 줄만 출력:\n"
       '  {"action":"ask","question":"..."} 또는 {"action":"flag","guess":"단어"}')

ASK_OPTS = ["네", "아니오", "가끔", "모름"]
FLAG_OPTS = ["네, 맞습니다", "아니오, 틀렸습니다"]

print(f"[로딩] {ADAPTER}  (첫 실행은 30초~1분)", flush=True)
model, tok = FastModel.from_pretrained(model_name=ADAPTER, max_seq_length=1024,
                                       load_in_4bit=True, device_map={"": 0})
model.eval()
inner = getattr(tok, "tokenizer", tok)


def next_action(hist):
    user = f"지금까지 기록:\n{hist or '(없음)'}\n\n다음 행동을 결정하세요."
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
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
        return advance(state)

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
            score = 0.85 ** max(state["turn"] - 5, 0)
            state["log"].append(f"→ 정답! {state['turn']}턴 만에 맞혔습니다 🎉 (S={score:.3f})")
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
        "마음속으로 단어 하나를 정하세요(학습 단어 목록: `data/words_917.txt`, 목록 밖 단어도 가능). "
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
