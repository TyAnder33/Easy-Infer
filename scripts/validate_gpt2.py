#!/usr/bin/env python3

import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel


if len(sys.argv) != 4:
    raise SystemExit(
        "usage: validate_gpt2.py <model-directory> <logits-file> <id,id,...>"
    )

model_directory, logits_path, token_text = sys.argv[1:]
token_ids = [int(token) for token in token_text.split(",")]

model = GPT2LMHeadModel.from_pretrained(
    model_directory, local_files_only=True
).to(device="cuda", dtype=torch.float32)
model.eval()

inputs = torch.tensor([token_ids], device="cuda", dtype=torch.long)
with torch.inference_mode():
    expected = model(inputs, use_cache=False).logits[0, -1].cpu().numpy()

actual = np.fromfile(logits_path, dtype=np.float32)
expected_count = 50_257
if expected.size != expected_count:
    raise SystemExit(f"wrong model vocabulary: {expected.size}, expected {expected_count}")
if actual.size != expected_count:
    raise SystemExit(f"wrong logits count: {actual.size}, expected {expected_count}")

difference = np.abs(actual - expected)
expected_token = int(expected.argmax())
actual_token = int(actual.argmax())

print(f"maximum error: {difference.max():.8g}")
print(f"mean error:    {difference.mean():.8g}")
print(f"HF token:      {expected_token}")
print(f"engine token:  {actual_token}")

if not np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
    raise SystemExit("logits do not match")
if actual_token != expected_token:
    raise SystemExit("greedy token does not match")

print("validation passed")
