import torch
from tqdm import tqdm

def eval_ppl(model, tokenizer, dataset, col="text", stride=512, max_len=1024, eval_frac=1.0):
    device = model.device

    texts = [t for t in dataset[col] if t.strip()]
    text = "\n\n".join(texts)

    print("Tokenizing evaluation dataset...")
    enc = tokenizer(text, return_tensors="pt")

    n_tokens = enc.input_ids.size(1)
    n_eval = n_tokens if eval_frac >= 1.0 else max(max_len, int(n_tokens * eval_frac))
    print(f"Total tokens for evaluation: {n_eval} of {n_tokens}")

    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, n_eval, stride), desc="Evaluating perplexity"):
        end = min(begin + max_len, n_eval)
        tgt_len = end - prev_end
        input_ids = enc.input_ids[:, begin:end].to(device)
        labels = input_ids.clone()
        labels[:, :-tgt_len] = -100

        with torch.no_grad():
            out = model(input_ids, labels=labels)
            nll = out.loss

        nlls.append(nll)

        prev_end = end
        if end == n_eval:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def make_probe(tokenizer, dataset, n_seq=4, seqlen=128, col="text"):
    texts = [t for t in dataset[col] if t.strip()]
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]

    chunks = []
    for i in range(n_seq):
        start = i * seqlen
        if start + seqlen <= ids.size(0):
            chunks.append(ids[start:start + seqlen])

    if not chunks:
        raise ValueError("Not enough tokens to build the divergence probe.")

    return torch.stack(chunks)


@torch.no_grad()
def dense_outputs(model, probe):
    device = model.device
    out = model(probe.to(device), output_hidden_states=True)

    logits = out.logits.reshape(-1, out.logits.size(-1)).float()
    logp = torch.log_softmax(logits, dim=-1)
    hidden = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1)).float()

    return {
        "logprobs": logp.half().cpu(),
        "argmax": logp.argmax(-1).cpu(),
        "hidden": hidden.half().cpu(),
    }


@torch.no_grad()
def compare_to_dense(model, probe, ref):
    device = model.device
    out = model(probe.to(device), output_hidden_states=True)

    logits = out.logits.reshape(-1, out.logits.size(-1)).float()
    logp = torch.log_softmax(logits, dim=-1)
    hidden = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1)).float()

    ref_logp = ref["logprobs"].float().to(device)
    ref_top = ref["argmax"].to(device)
    ref_h = ref["hidden"].float().to(device)

    p_ref = ref_logp.exp()
    kl = (p_ref * (ref_logp - logp)).sum(-1).mean().item()

    top1 = (logp.argmax(-1) == ref_top).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(hidden, ref_h, dim=-1).mean().item()

    return {"kl_dense_pruned": kl, "top1_agreement": top1, "hidden_cosine": cos}
