# Third-party material

| Path | Source | License |
|---|---|---|
| `src/qos_orchestration/ifeval/` | IFEval checkers by Google Research, via [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (commit `d6de816`) | Apache License 2.0 (`src/qos_orchestration/ifeval/LICENSE`) |
| `data/tasks.jsonl` (`qa-*`) | [NQ-open](https://huggingface.co/datasets/google-research-datasets/nq_open), questions and answers | CC BY-SA 3.0 |
| `data/tasks.jsonl` (`math-*`) | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), problems and answers | MIT |
| `data/tasks.jsonl` (`code-*`) | [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), prompts and tests | MIT |
| `data/tasks.jsonl` (`long-*`) | [IFEval](https://huggingface.co/datasets/google/IFEval), prompts and instruction arguments | Apache License 2.0 |
| `results/raw/`, `results/scored_*.jsonl` | outputs of Qwen2.5-3B/7B-Instruct (local) and Qwen2.5-72B-Instruct (OpenRouter) on these tasks | see the Qwen2.5 model cards for the model licenses |

Dataset revisions and the sampled row indices are recorded in `data/datasets_manifest.json`.

## License notices

GSM8K and HumanEval are distributed under the MIT License:

> Copyright (c) 2021 OpenAI
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
> Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
> WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

NQ-open is distributed under the Creative Commons Attribution-ShareAlike 3.0 license
(https://creativecommons.org/licenses/by-sa/3.0/); the `qa-*` items in `data/tasks.jsonl` are shared under the same
license. IFEval and the IFEval checkers are distributed under the Apache License 2.0 (`src/qos_orchestration/ifeval/LICENSE`).
