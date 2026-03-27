#!/usr/bin/env python3
"""
QuantumGPT inference script.
Loads QuantumGPT-124M-v2 from HuggingFace and generates OpenQASM 2.0 circuits
from natural language descriptions.

Model: merileijona/quantumgpt-124m-v2 (default)
"""
import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "<|user|>{description}<|end|>\n<|assistant|>"
STOP_TEXT = "<|end|>"


def load_model(model_path: str):
    """Load model and tokenizer from HF repo ID or local path."""
    print(f"Loading model: {model_path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # set to inference mode (PyTorch nn.Module method)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def generate_circuit(model, tok, prompt: str, temperature: float = 0.8) -> str:
    """Generate a single QASM circuit for the given prompt."""
    device = next(model.parameters()).device
    formatted = PROMPT_TEMPLATE.format(description=prompt)
    inputs = tok(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(output[0], skip_special_tokens=False)
    assistant_part = decoded[len(formatted):]

    if STOP_TEXT in assistant_part:
        assistant_part = assistant_part.split(STOP_TEXT, 1)[0]

    return assistant_part.strip()


def validate_qasm(qasm: str) -> bool:
    """Parse QASM with qiskit. Returns True if valid."""
    try:
        from qiskit.qasm2 import loads
        loads(qasm)
        return True
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Generate OpenQASM 2.0 circuits from natural language descriptions."
    )
    ap.add_argument(
        "--prompt", required=True, type=str,
        help='Natural language description, e.g. "Create a Bell state with two qubits"'
    )
    ap.add_argument(
        "--model", default="merileijona/quantumgpt-124m-v2", type=str,
        help="HuggingFace repo ID or local path to HF model directory "
             "(default: merileijona/quantumgpt-124m-v2)"
    )
    ap.add_argument(
        "--samples", type=int, default=1,
        help="Number of circuits to generate (default: 1)"
    )
    ap.add_argument(
        "--temp", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    ap.add_argument(
        "--validate", action="store_true",
        help="Run qiskit QASM parser on output and report pass/fail"
    )
    args = ap.parse_args()

    model, tok = load_model(args.model)

    for i in range(args.samples):
        if args.samples > 1:
            print(f"--- Sample {i+1}/{args.samples} ---", file=sys.stderr)

        qasm = generate_circuit(model, tok, args.prompt, temperature=args.temp)
        print(qasm)

        if args.validate:
            ok = validate_qasm(qasm)
            status = "PASS" if ok else "FAIL"
            print(f"[validate: {status}]", file=sys.stderr)

        if i < args.samples - 1:
            print()


if __name__ == "__main__":
    main()
