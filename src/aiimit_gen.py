"""Generate AI imitations with an open model (vLLM) from the frozen prompt table.
Usage: python aiimit_gen.py <HF repo> <short name> [--n 3] [--gpu_mem 0.9] [--max_len 2048]
Protocol: chat template, user turn only, thinking off where supported, temperature 0.7, top_p 0.9, n samples per prompt.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import argparse, glob, json, os, time
import pandas as pd

P = (PROJECT + '/data')


def snapshot(repo):
    d = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub', 'models--' + repo.replace('/', '--'), 'snapshots')
    return sorted(glob.glob(d + '/*'), key=os.path.getmtime)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('name')
    ap.add_argument('--n', type=int, default=3); ap.add_argument('--gpu_mem', type=float, default=0.9)
    ap.add_argument('--max_len', type=int, default=2048); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = f'{P}/aiimit/out_{a.name}.jsonl'
    if os.path.exists(out) and not a.limit:
        print('exists, skip', out); return
    from vllm import LLM, SamplingParams
    pr = pd.read_csv(f'{P}/aiimit_prompts_v0.tsv', sep='\t')
    if a.limit: pr = pr.head(a.limit)
    llm = LLM(snapshot(a.model), max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    kw = {'enable_thinking': False} if 'enable_thinking' in (getattr(tok, 'chat_template', None) or '') else {}
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, **kw)
             for p in pr.prompt]
    print('RENDERED EXAMPLE:\n' + texts[0], flush=True)
    sp = SamplingParams(n=a.n, temperature=0.7, top_p=0.9, max_tokens=a.max_len - 256, seed=0)
    t0 = time.time()
    res = llm.generate(texts, sp)
    path = out if not a.limit else f'/tmp/aiimit_smoke_{a.name}.jsonl'
    with open(path, 'w', encoding='utf8') as f:
        for i, o in zip(pr.id, res):
            for c in o.outputs:
                f.write(json.dumps(dict(id=i, text=c.text.strip(), model=a.model), ensure_ascii=False) + '\n')
    print(f'wrote {path} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
