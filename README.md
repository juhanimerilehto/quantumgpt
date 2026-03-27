# QuantumGPT-124M

![QuantumGPT](quantumgpt.svg)



```powershell
python generate.py --prompt "Create a Bell state with two qubits"
```

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
```

A 124-million-parameter language model trained from scratch on synthetic quantum circuit data. Give it a description in plain English. It returns valid OpenQASM 2.0. This is the terminal point of the pipeline: the interface between human intent and quantum gate sequences. The weights encode a grammar of gates; the prompt activates it.

This model was built as part of a data scaling study for quantum circuit language modeling. The full methodology is described in: *QuantumGPT: A Data Scaling Study for Quantum Circuit Generation, Merilehto 2026*.

---

## Requirements

- Python 3.11
- PyTorch >= 2.0
- `transformers` >= 4.35
- Internet connection (model downloaded from HuggingFace on first run, ~500MB)

Optional: `qiskit>=1.0` for `--validate` flag

## Installation

```bash
conda env create -f environment.yml
conda activate quantumgpt
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Usage

```powershell
# Generate one circuit
python generate.py --prompt "Create a Bell state with two qubits"

# Generate multiple samples
python generate.py --prompt "Implement Grover's algorithm" --samples 3

# Validate output with qiskit (requires: pip install qiskit)
python generate.py --prompt "Apply Hadamard gate" --validate

# Use a local model directory
python generate.py --model ./hf_v2 --prompt "Apply Hadamard gate"

# Higher temperature for more diversity
python generate.py --prompt "Quantum teleportation" --samples 5 --temp 1.0
```

Arguments:
```
--prompt TEXT    Natural language description (required)
--model PATH     HF repo ID or local path (default: merileijona/quantumgpt-124m-v2)
--samples N      Number of circuits to generate (default: 1)
--temp FLOAT     Sampling temperature (default: 0.8)
--validate       Parse output with qiskit and report PASS/FAIL
```

---

## Model

- Architecture: GPT-2 small (12L, 12H, 768E, 256 ctx, ~123.8M params)
- Tokenizer: GPT-2 BPE (tiktoken gpt2, vocab padded to 50304)
- Training format: `<|user|>{description}<|end|>\n<|assistant|>{qasm}<|end|>`
- Hosted at: `merileijona/quantumgpt-124m-v2`

---

## Pipeline

This repo is the inference end of a three-stage pipeline:

```
quantum-circuit-training-data-generator  ->  quantumgpt-training  ->  quantumgpt (this repo)
```

---

## Hardware Notes

Inference runs on CPU or GPU. The model is ~500MB. On an RTX 4070, generation takes under a second per circuit.

---

## License

MIT
