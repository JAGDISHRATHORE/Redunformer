import torch
from tqdm import tqdm

def evaluate_perplexity(model, tokenizer, dataset, text_column="text", stride=512, max_length=1024):
    device = model.device
    
    # Filter out empty strings if any
    texts = [t for t in dataset[text_column] if t.strip()]
    full_text = "\n\n".join(texts)
    
    print("Tokenizing evaluation dataset...")
    encodings = tokenizer(full_text, return_tensors="pt")
    
    seq_len = encodings.input_ids.size(1)
    print(f"Total tokens for evaluation: {seq_len}")
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
