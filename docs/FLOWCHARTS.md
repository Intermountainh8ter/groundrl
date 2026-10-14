# Flowcharts

Both charts are Mermaid sources kept in `docs/diagrams/*.mmd`; the rendered SVGs next to them are
regenerated with `scripts/render_diagrams.sh` (needs node 18+ and a Chromium; set
`PUPPETEER_EXECUTABLE_PATH` to use an existing browser).

## Usage: from corpus to report

Blue nodes run on a laptop; yellow nodes need a GPU and have never been run in this repository.

![Usage flowchart](diagrams/usage.svg)

```mermaid
flowchart TB
    subgraph prep ["Prepare (CPU, no GPU needed)"]
        direction LR
        A["1. Fetch corpus<br/>scripts/fetch_corpus.py<br/>130 EIPs + sha256"] --> B["2. Build tasks<br/>groundrl build-tasks<br/>1792 tasks, split by EIP"]
        B --> C["3. Audit the reward<br/>scripts/reward_audit.py<br/>19 policies x v0 v1 v2"]
        C --> D{"Any hack at or<br/>above 0.50<br/>under v2?"}
        D -- "yes" --> E["Revise rewards.py<br/>then re-audit"]
        E --> C
        D -- "no" --> F["Scripted baselines<br/>groundrl baseline-eval"]
    end
    subgraph train ["Train (GPU, never run in this repository)"]
        direction LR
        G["4. SFT baseline<br/>train/sft.py<br/>LoRA, then merge"] --> H["5. GRPO, reward v2<br/>train/grpo.py<br/>--init-from merged"]
        H --> I{"verified rate up,<br/>correctness down?"}
        I -- "yes: hacking" --> J["Stop, keep last good<br/>checkpoint, extend audit"]
        I -- "no" --> K["6. Evaluate on TEST<br/>train/eval.py<br/>3+ seeds"]
    end
    subgraph out ["Report"]
        direction LR
        L["7. make_report.py<br/>renders results/*.json only"] --> M{"Beats the extractive<br/>baseline with CI margin?"}
        M -- "yes" --> N["Report with intervals<br/>and the threats list"]
        M -- "no" --> O["Report it anyway:<br/>no gain over locate-and-copy"]
    end
    prep --> train --> out

    classDef gpu fill:#fde68a,stroke:#92400e,color:#1f2937
    classDef cpu fill:#bfdbfe,stroke:#1e3a8a,color:#1f2937
    classDef decision fill:#e5e7eb,stroke:#374151,color:#1f2937
    class A,B,C,E,F,L cpu
    class G,H,K gpu
    class D,I,M decision
```

## Troubleshooting: decision tree on real symptoms

Every leaf is a concrete change. The metric names are the ones TRL logs plus the `groundrl/*`
diagnostics added by the reward wrapper (`make_trl_reward`); see `docs/RUNBOOK.md` section 5.

![Troubleshooting flowchart](diagrams/troubleshooting.svg)

```mermaid
flowchart LR
    S{"What do the GRPO logs show?"}
    S --> R0["Reward flat at zero"]
    S --> R1["Reward spikes, then collapses"]
    S --> R2["KL blows up"]
    S --> R3["Out of memory"]
    S --> R4["Completions all abstain"]
    S --> R5["Citation format drifts"]
    S --> R6["Eval much worse than train"]

    R0 --> R0a{"groundrl/verified_rate<br/>is also 0?"}
    R0a -- "yes" --> R0b["The model never emits valid citations.<br/>Fix: run SFT first and start GRPO<br/>from outputs/sft/merged (--init-from)"]
    R0a -- "no" --> R0c{"frac_reward_zero_std<br/>near 1.0?"}
    R0c -- "yes" --> R0d["Every group scores the same.<br/>Fix: raise group size to 8,<br/>keep temperature at 1.0, check floor 0.10 is on"]
    R0c -- "no" --> R0e{"completions/clipped_ratio<br/>high?"}
    R0e -- "yes" --> R0f["Answers are cut off before the citation.<br/>Fix: raise max_completion_length and<br/>data.max_completion_tokens together"]
    R0e -- "no" --> R0g["Check KeyError example_json:<br/>remove_unused_columns must be false"]

    R1 --> R1a{"verified_rate up while<br/>groundrl/correctness down?"}
    R1a -- "yes" --> R1b["Reward hacking: grounded but wrong.<br/>Fix: stop, keep the last good checkpoint,<br/>read logged completions, add the exploit<br/>to audit.py, revise the reward, re-audit"]
    R1a -- "no" --> R1c{"mean_words growing?"}
    R1c -- "yes" --> R1d["Length drift. Fix: confirm the length term,<br/>lower learning rate to 5e-6"]
    R1c -- "no" --> R1e["Policy collapse. Fix: raise beta to 0.05,<br/>lower learning rate, lower steps_per_generation"]

    R2 --> R2a{"Started from merged SFT model?"}
    R2a -- "no" --> R2b["KL is measured to the base model.<br/>Fix: pass --init-from outputs/sft/merged"]
    R2a -- "yes" --> R2c["Fix, in order: learning rate 2e-5 to 5e-6,<br/>beta 0.02 to 0.05, max_grad_norm 0.5,<br/>keep num_iterations at 1"]

    R3 --> R3a{"Where does it fail?"}
    R3a -- "while generating" --> R3b["Fix: lower steps_per_generation or<br/>per_device_train_batch_size, lower<br/>max_prompt_tokens, try use_vllm colocate on Linux"]
    R3a -- "in the training step" --> R3c["Fix: per_device_train_batch_size 1,<br/>gradient_checkpointing true, 4-bit profile.<br/>The 151,936-token vocabulary logits dominate"]
    R3a -- "in eval" --> R3d["Fix: eval.batch_size 1 to 2.<br/>Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]

    R4 --> R4a{"abstain_rate above 0.5<br/>and answers score below 0.11?"}
    R4a -- "yes" --> R4b["Refusing pays more than attempting.<br/>Fix: more SFT epochs so attempts earn reward,<br/>check prompts really contain the gold chunk,<br/>lower learning rate"]
    R4a -- "no" --> R4c["Check data: unanswerable share should be<br/>about 11 percent in the train split"]

    R5 --> R5a{"format_drift_rate above 0?"}
    R5a -- "yes" --> R5b["Wrong shape such as eip-1559 hash 7 or capitals.<br/>Fix: SFT longer, same chat template at train<br/>and eval, eval temperature 0"]
    R5a -- "no" --> R5c{"hallucinated_citation_rate high?"}
    R5c -- "yes" --> R5d["Cites chunks that are not in the context.<br/>Fix: lower sampling temperature,<br/>more SFT, check the prompt lists the ids"]
    R5c -- "no" --> R5e["Check suspicious_character_rate:<br/>tokenizer or locale issue on Windows (PYTHONUTF8=1)"]

    R6 --> R6a{"Same prompt rendering and<br/>same decoding?"}
    R6a -- "no" --> R6b["Fix: use the shared prompts.py path,<br/>enable_thinking false in both,<br/>compare at the same temperature"]
    R6a -- "yes" --> R6c{"Gap only on TEST, not VAL?"}
    R6c -- "yes" --> R6d["Few TEST EIPs: wide intervals.<br/>Fix: report bootstrap CIs, add seeds,<br/>do not tune on TEST"]
    R6c -- "no" --> R6e["Generalisation across EIPs is the real gap.<br/>Check train prompts dropped by the budget<br/>and the extractive baseline on both splits"]

    classDef start fill:#e5e7eb,stroke:#374151,color:#1f2937
    classDef fix fill:#bbf7d0,stroke:#166534,color:#1f2937
    classDef ask fill:#bfdbfe,stroke:#1e3a8a,color:#1f2937
    class S start
    class R0b,R0d,R0f,R0g,R1b,R1d,R1e,R2b,R2c,R3b,R3c,R3d,R4b,R4c,R5b,R5d,R5e,R6b,R6d,R6e fix
    class R0a,R0c,R0e,R1a,R1c,R2a,R3a,R4a,R5a,R5c,R6a,R6c ask
```
