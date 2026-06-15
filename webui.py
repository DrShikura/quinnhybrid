#!/usr/bin/env python3
"""
Mobile-friendly Flask UI for testing WaveGPT inference.

Serves on the local network (default 0.0.0.0:5007). Lets you:
  - pick a trained checkpoint
  - tweak sampling settings (temperature / top-k / top-p / mode) or use presets
  - generate a continuation and tap any generated token to see the top-10
    predicted tokens (with probabilities) at that step, before sampling
  - run a small suite of basic prompt tests

Run:
    CUDA_VISIBLE_DEVICES=1 python webui.py                      # pin to one GPU
    python webui.py --host 0.0.0.0 --port 5007 --device cpu
"""
import argparse
import sys
import time
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from spectral.experiments.infer import load_from_checkpoint
from spectral.model.generate import _top_k_top_p_filter, generate_beam

app = Flask(__name__)

# Resolved at startup
DEVICE = "cpu"
_MODELS = {}            # checkpoint_path -> (model, tokenizer, config)
_LOCK = threading.Lock()

_TERMINALS = {"<EOS>", "ENDMARKER", "NEWLINE", "NL"}
TEST_PROMPTS = ["def ", "class ", "import ", "for i in range(", "if __name__", "return "]


# ── Model handling ─────────────────────────────────────────────────────────
def discover_checkpoints():
    out = []
    for p in sorted(REPO.glob("checkpoints/*/spectral_best.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = ck.get("config", {})
            vl = ck.get("val_loss")
            out.append({
                "path": str(p.relative_to(REPO)),
                "name": p.parent.name,
                "model_type": cfg.get("model_type", "?"),
                "tokenizer": cfg.get("tokenizer_type", "structural"),
                "d_model": cfg.get("d_model", cfg.get("embed_dim")),
                "n_layers": cfg.get("n_layers"),
                "epoch": ck.get("epoch"),
                "val_loss": round(vl, 4) if isinstance(vl, (int, float)) else None,
            })
        except Exception as e:
            out.append({"path": str(p.relative_to(REPO)), "name": p.parent.name,
                        "error": str(e)})
    return out


def get_model(ckpt_path: str):
    """Load (and cache) a checkpoint, moved to DEVICE."""
    full = str((REPO / ckpt_path).resolve())
    if full not in _MODELS:
        model, tokenizer, config = load_from_checkpoint(full, "auto")
        model.to(DEVICE)
        model.eval()
        _MODELS[full] = (model, tokenizer, config)
    return _MODELS[full]


def _strip_terminals(ids, tokenizer):
    while len(ids) > 1 and tokenizer.vocab[ids[-1]] in _TERMINALS:
        ids = ids[:-1]
    return ids


def _name(tokenizer, i):
    return tokenizer.vocab[i] if 0 <= i < len(tokenizer.vocab) else "<UNK>"


@torch.no_grad()
def run_generate(ckpt, prompt, max_new, temperature, top_k, top_p, mode, n_beams):
    model, tokenizer, config = get_model(ckpt)
    max_seq = getattr(model, "max_seq_len", 256)

    if prompt.strip():
        ids = tokenizer.encode(prompt, add_special=True)
        ids = _strip_terminals(ids, tokenizer)
    else:
        ids = [tokenizer.bos_id]

    prompt_tokens = [_name(tokenizer, i) for i in ids]
    t0 = time.time()

    # Beam search: no per-token distributions
    if mode == "beam":
        text = generate_beam(model, tokenizer, prompt=prompt, max_new=max_new,
                             n_beams=n_beams, device=DEVICE)
        dt = time.time() - t0
        return {"ok": True, "mode": "beam", "prompt_tokens": prompt_tokens,
                "steps": [], "text": text, "elapsed_ms": round(dt * 1000),
                "n_new": None, "tok_per_sec": None,
                "note": "Beam search returns the single best path; per-token "
                        "top-10 inspection is only available in sample/greedy mode."}

    generated = list(ids)
    steps = []
    for _ in range(max_new):
        ctx = generated[-max_seq:]
        tok = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        pos = torch.arange(len(ctx), dtype=torch.long, device=DEVICE).unsqueeze(0)
        out = model(tok, pos, mode="lm")
        logits = out["lm_logits"][0, -1, :]

        raw = F.softmax(logits, dim=-1)            # distribution BEFORE sampling
        tp, ti = raw.topk(min(10, raw.numel()))
        top10 = [{"token": _name(tokenizer, int(i)), "id": int(i),
                  "prob": float(p)} for p, i in zip(tp.tolist(), ti.tolist())]

        if mode == "greedy" or temperature <= 0:
            nxt = int(logits.argmax().item())
        else:
            l = logits / temperature
            l = _top_k_top_p_filter(l, top_k=top_k, top_p=top_p)
            probs = F.softmax(l, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())

        steps.append({"token": _name(tokenizer, nxt), "id": nxt,
                      "chosen_prob": float(raw[nxt].item()), "top10": top10})
        generated.append(nxt)
        if nxt == tokenizer.eos_id:
            break

    dt = time.time() - t0
    text = " ".join(tokenizer.decode(generated))
    n_new = len(steps)
    return {"ok": True, "mode": mode, "prompt_tokens": prompt_tokens, "steps": steps,
            "text": text, "elapsed_ms": round(dt * 1000), "n_new": n_new,
            "tok_per_sec": round(n_new / dt, 1) if dt > 0 else None}


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/info")
def api_info():
    return jsonify({"device": DEVICE,
                    "cuda": torch.cuda.is_available(),
                    "gpu": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None)})


@app.route("/api/checkpoints")
def api_checkpoints():
    return jsonify({"checkpoints": discover_checkpoints()})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    d = request.get_json(force=True)
    try:
        with _LOCK:
            res = run_generate(
                ckpt=d["checkpoint"],
                prompt=d.get("prompt", ""),
                max_new=int(d.get("max_new", 60)),
                temperature=float(d.get("temperature", 0.8)),
                top_k=int(d.get("top_k", 50)),
                top_p=float(d.get("top_p", 0.95)),
                mode=d.get("mode", "sample"),
                n_beams=int(d.get("n_beams", 4)),
            )
        return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/test", methods=["POST"])
def api_test():
    d = request.get_json(force=True)
    ckpt = d["checkpoint"]
    try:
        with _LOCK:
            model, tokenizer, config = get_model(ckpt)
            info = {
                "params": model.param_count().get("total"),
                "vocab_size": len(tokenizer.vocab),
                "model_type": config.get("model_type"),
                "tokenizer": config.get("tokenizer_type", "structural"),
                "device": DEVICE,
            }
            results = []
            for pr in TEST_PROMPTS:
                r = run_generate(ckpt, pr, max_new=40, temperature=0.8,
                                 top_k=50, top_p=0.95, mode="sample", n_beams=4)
                results.append({"prompt": pr, "text": r["text"],
                                "n_new": r["n_new"], "ms": r["elapsed_ms"],
                                "tok_per_sec": r["tok_per_sec"]})
        return jsonify({"ok": True, "info": info, "results": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#0f1117">
<title>WaveGPT Playground</title>
<style>
:root{--bg:#0f1117;--panel:#171a23;--panel2:#1f2430;--line:#2a3040;--fg:#e6e9ef;--mut:#8b93a7;--acc:#5b9dff;--acc2:#7c5cff;--good:#3ecf8e;--warn:#ffb454;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:760px;margin:0 auto;padding:14px 12px 64px}
h1{font-size:18px;margin:4px 0 2px;display:flex;align-items:center;gap:8px}
h1 .dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 8px var(--good)}
.sub{color:var(--mut);font-size:12px;margin-bottom:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:12px}
label{display:block;font-size:12px;color:var(--mut);margin:8px 0 4px;text-transform:uppercase;letter-spacing:.04em}
select,textarea,input[type=text]{width:100%;background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:11px;font-size:15px;font-family:inherit}
textarea{min-height:70px;resize:vertical;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.row{display:flex;gap:8px;flex-wrap:wrap}
.row>div{flex:1;min-width:120px}
.slider{display:flex;align-items:center;gap:10px}
input[type=range]{flex:1;accent-color:var(--acc)}
.val{min-width:42px;text-align:right;font-variant-numeric:tabular-nums;color:var(--fg);font-size:14px}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.chip{background:var(--panel2);border:1px solid var(--line);color:var(--fg);border-radius:999px;padding:8px 14px;font-size:13px;cursor:pointer;user-select:none}
.chip:active{transform:scale(.96)}
.chip.sel{border-color:var(--acc);background:#17233b;color:#cfe0ff}
.btn{width:100%;border:0;border-radius:12px;padding:14px;font-size:16px;font-weight:600;cursor:pointer;color:#fff;background:linear-gradient(90deg,var(--acc),var(--acc2));margin-top:10px}
.btn:active{transform:scale(.99)}
.btn.sec{background:var(--panel2);border:1px solid var(--line);color:var(--fg);font-weight:500}
.btn[disabled]{opacity:.5}
.meta{font-size:12px;color:var(--mut);margin-top:8px;display:flex;gap:12px;flex-wrap:wrap}
.meta b{color:var(--fg);font-weight:600}
.out{margin-top:6px;line-height:2.1}
.tok{display:inline-block;padding:2px 6px;margin:2px 2px;border-radius:7px;font-family:ui-monospace,Menlo,monospace;font-size:13px;white-space:pre}
.tok.p{background:#222838;color:var(--mut)}
.tok.g{background:#1d2a44;color:#dbe6ff;cursor:pointer;border:1px solid #2c3e63}
.tok.g:active{transform:scale(.95)}
.tok.eos{background:#3a2030;color:#ffb0c8}
pre.txt{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px;white-space:pre-wrap;word-break:break-word;font-size:12px;color:var(--mut);max-height:160px;overflow:auto}
.hint{font-size:12px;color:var(--mut);margin-top:6px}
.tlist{margin-top:8px}
.trow{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--line)}
.trow .nm{flex:0 0 38%;font-family:ui-monospace,Menlo,monospace;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.trow .bar{flex:1;height:14px;background:var(--panel);border-radius:7px;overflow:hidden}
.trow .bar > i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.trow .pc{flex:0 0 56px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px;color:var(--mut)}
.trow.sel .nm{color:var(--good);font-weight:700}
.trow.sel .bar>i{background:linear-gradient(90deg,var(--good),#2bb)}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-end;z-index:20}
.modal.on{display:flex}
.sheet{background:var(--panel);border-top-left-radius:18px;border-top-right-radius:18px;border-top:1px solid var(--line);width:100%;max-width:760px;margin:0 auto;padding:14px 14px 24px;max-height:80vh;overflow:auto}
.sheet h3{margin:2px 0 2px;font-size:15px}
.sheet .x{float:right;color:var(--mut);font-size:22px;line-height:1;cursor:pointer;padding:0 6px}
.tests .tr{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 10px;margin-top:8px}
.tests .tp{font-family:ui-monospace,Menlo,monospace;color:var(--acc);font-size:13px}
.tests .to{font-family:ui-monospace,Menlo,monospace;color:var(--fg);font-size:12px;margin-top:4px;white-space:pre-wrap;word-break:break-word}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #fff5;border-top-color:#fff;border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
.err{color:#ff8b8b;font-size:13px;margin-top:8px}
</style>
</head>
<body>
<div class="wrap">
  <h1><span class="dot"></span>WaveGPT Playground</h1>
  <div class="sub" id="devline">loading…</div>

  <div class="card">
    <label>Checkpoint</label>
    <select id="ckpt"></select>
    <div class="meta" id="ckmeta"></div>
  </div>

  <div class="card">
    <label>Prompt</label>
    <textarea id="prompt" placeholder="def ">def </textarea>

    <label>Presets</label>
    <div class="presets" id="presets">
      <div class="chip" data-p="greedy">Greedy</div>
      <div class="chip sel" data-p="balanced">Balanced</div>
      <div class="chip" data-p="creative">Creative</div>
      <div class="chip" data-p="precise">Precise</div>
      <div class="chip" data-p="beam">Beam</div>
    </div>

    <label>Mode</label>
    <select id="mode">
      <option value="sample">Sample</option>
      <option value="greedy">Greedy (argmax)</option>
      <option value="beam">Beam search</option>
    </select>

    <div class="row">
      <div>
        <label>Max new tokens</label>
        <div class="slider"><input type="range" id="max_new" min="5" max="256" value="60"><span class="val" id="max_new_v">60</span></div>
      </div>
      <div>
        <label>Temperature</label>
        <div class="slider"><input type="range" id="temperature" min="0" max="2" step="0.05" value="0.8"><span class="val" id="temperature_v">0.80</span></div>
      </div>
    </div>
    <div class="row">
      <div>
        <label>Top-k (0=off)</label>
        <div class="slider"><input type="range" id="top_k" min="0" max="200" value="50"><span class="val" id="top_k_v">50</span></div>
      </div>
      <div>
        <label>Top-p</label>
        <div class="slider"><input type="range" id="top_p" min="0.1" max="1" step="0.01" value="0.95"><span class="val" id="top_p_v">0.95</span></div>
      </div>
    </div>
    <div class="row" id="beamrow" style="display:none">
      <div>
        <label>Beams</label>
        <div class="slider"><input type="range" id="n_beams" min="2" max="12" value="4"><span class="val" id="n_beams_v">4</span></div>
      </div>
    </div>

    <button class="btn" id="gen">Generate</button>
    <button class="btn sec" id="runtests">Run basic tests</button>
  </div>

  <div class="card" id="outcard" style="display:none">
    <label>Output <span style="text-transform:none;color:var(--acc)">— tap a blue token for top-10</span></label>
    <div class="out" id="out"></div>
    <div class="meta" id="outmeta"></div>
    <div class="hint" id="outnote"></div>
    <label style="margin-top:10px">Reconstructed</label>
    <pre class="txt" id="rawtext"></pre>
  </div>

  <div class="card tests" id="testcard" style="display:none">
    <label>Basic tests</label>
    <div class="meta" id="testinfo"></div>
    <div id="testout"></div>
  </div>

  <div class="err" id="err"></div>
</div>

<div class="modal" id="modal"><div class="sheet">
  <span class="x" id="mclose">×</span>
  <h3 id="mtitle">Top-10</h3>
  <div class="hint" id="msub"></div>
  <div class="tlist" id="mlist"></div>
</div></div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let LAST=null;

const PRESETS={
 greedy:{mode:'greedy',temperature:0,top_k:0,top_p:1},
 balanced:{mode:'sample',temperature:0.8,top_k:50,top_p:0.95},
 creative:{mode:'sample',temperature:1.1,top_k:0,top_p:0.98},
 precise:{mode:'sample',temperature:0.5,top_k:20,top_p:0.9},
 beam:{mode:'beam'}
};

function bind(id){const r=$('#'+id),v=$('#'+id+'_v');const f=()=>{let x=r.value;if(id=='temperature'||id=='top_p')x=(+x).toFixed(2);v.textContent=x;};r.addEventListener('input',f);f();}
['max_new','temperature','top_k','top_p','n_beams'].forEach(bind);

function setVals(o){for(const k in o){if(k=='mode'){$('#mode').value=o[k];continue;}const el=$('#'+k);if(el){el.value=o[k];el.dispatchEvent(new Event('input'));}}syncMode();}
$('#presets').addEventListener('click',e=>{const c=e.target.closest('.chip');if(!c)return;$$('#presets .chip').forEach(x=>x.classList.remove('sel'));c.classList.add('sel');setVals(PRESETS[c.dataset.p]);});
function syncMode(){$('#beamrow').style.display=$('#mode').value=='beam'?'flex':'none';}
$('#mode').addEventListener('change',syncMode);

async function jget(u){const r=await fetch(u);return r.json();}
async function jpost(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json();}

async function loadCkpts(){
  const info=await jget('/api/info');
  $('#devline').textContent='device: '+info.device+(info.gpu?(' · '+info.gpu):'');
  const d=await jget('/api/checkpoints');const sel=$('#ckpt');sel.innerHTML='';
  d.checkpoints.forEach(c=>{const o=document.createElement('option');o.value=c.path;
    o.textContent=c.name+(c.val_loss!=null?(' · val '+c.val_loss):'');sel.appendChild(o);o._m=c;});
  showMeta();
}
function showMeta(){const o=$('#ckpt').selectedOptions[0];if(!o||!o._m)return;const c=o._m;
  $('#ckmeta').innerHTML=`<span>type <b>${c.model_type}</b></span><span>tok <b>${c.tokenizer}</b></span>`+
   `<span>d <b>${c.d_model}</b></span><span>layers <b>${c.n_layers}</b></span>`+
   `<span>epoch <b>${c.epoch}</b></span><span>val <b>${c.val_loss}</b></span>`;}
$('#ckpt').addEventListener('change',showMeta);

function settings(){return{checkpoint:$('#ckpt').value,prompt:$('#prompt').value,
  max_new:+$('#max_new').value,temperature:+$('#temperature').value,top_k:+$('#top_k').value,
  top_p:+$('#top_p').value,mode:$('#mode').value,n_beams:+$('#n_beams').value};}

function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function render(r){
  LAST=r;$('#outcard').style.display='block';$('#err').textContent='';
  const out=$('#out');out.innerHTML='';
  r.prompt_tokens.forEach(t=>{const s=document.createElement('span');s.className='tok p';s.textContent=t;out.appendChild(s);});
  r.steps.forEach((st,i)=>{const s=document.createElement('span');
    s.className='tok g'+(st.token=='<EOS>'?' eos':'');s.textContent=st.token;
    s.style.opacity=(0.55+0.45*Math.min(1,st.chosen_prob)).toFixed(2);
    s.dataset.i=i;s.addEventListener('click',()=>openTok(i));out.appendChild(s);});
  const m=[];if(r.n_new!=null)m.push('<span><b>'+r.n_new+'</b> tokens</span>');
  m.push('<span><b>'+r.elapsed_ms+'</b> ms</span>');
  if(r.tok_per_sec)m.push('<span><b>'+r.tok_per_sec+'</b> tok/s</span>');
  m.push('<span>mode <b>'+r.mode+'</b></span>');
  $('#outmeta').innerHTML=m.join('');
  $('#outnote').textContent=r.note||'';
  $('#rawtext').textContent=r.text;
}

function openTok(i){const st=LAST.steps[i];if(!st)return;
  $('#mtitle').textContent='Step '+(i+1)+': “'+st.token+'”';
  $('#msub').textContent='chosen prob '+(st.chosen_prob*100).toFixed(1)+'% · top-10 before sampling';
  const L=$('#mlist');L.innerHTML='';
  st.top10.forEach(c=>{const sel=c.id==st.id;const row=document.createElement('div');row.className='trow'+(sel?' sel':'');
    row.innerHTML=`<div class="nm">${esc(c.token)}</div><div class="bar"><i style="width:${(c.prob*100).toFixed(1)}%"></i></div><div class="pc">${(c.prob*100).toFixed(1)}%</div>`;
    L.appendChild(row);});
  $('#modal').classList.add('on');
}
$('#mclose').addEventListener('click',()=>$('#modal').classList.remove('on'));
$('#modal').addEventListener('click',e=>{if(e.target.id=='modal')$('#modal').classList.remove('on');});

$('#gen').addEventListener('click',async()=>{
  const b=$('#gen');b.disabled=true;const old=b.innerHTML;b.innerHTML='<span class="spin"></span> generating';
  $('#err').textContent='';
  try{const r=await jpost('/api/generate',settings());if(r.ok)render(r);else $('#err').textContent='Error: '+r.error;}
  catch(e){$('#err').textContent='Error: '+e;}
  b.disabled=false;b.innerHTML=old;
});

$('#runtests').addEventListener('click',async()=>{
  const b=$('#runtests');b.disabled=true;const old=b.innerHTML;b.innerHTML='<span class="spin"></span> running tests';
  $('#err').textContent='';
  try{const r=await jpost('/api/test',{checkpoint:$('#ckpt').value});
    if(!r.ok){$('#err').textContent='Error: '+r.error;}else{
      $('#testcard').style.display='block';const I=r.info;
      $('#testinfo').innerHTML=`<span>params <b>${I.params}</b></span><span>vocab <b>${I.vocab_size}</b></span>`+
        `<span>type <b>${I.model_type}</b></span><span>tok <b>${I.tokenizer}</b></span><span>dev <b>${I.device}</b></span>`;
      const o=$('#testout');o.innerHTML='';
      r.results.forEach(t=>{const d=document.createElement('div');d.className='tr';
        d.innerHTML=`<div class="tp">${esc(t.prompt)}</div><div class="to">${esc(t.text)}</div>`+
          `<div class="meta"><span>${t.n_new} tok</span><span>${t.ms} ms</span><span>${t.tok_per_sec||'-'} tok/s</span></div>`;
        o.appendChild(d);});
    }}
  catch(e){$('#err').textContent='Error: '+e;}
  b.disabled=false;b.innerHTML=old;
});

syncMode();loadCkpts();
</script>
</body>
</html>"""


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    print(f"Device: {DEVICE}")
    print(f"Checkpoints: {[c['name'] for c in discover_checkpoints()]}")
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
