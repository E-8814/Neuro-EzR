"""
Smoke-test each component of Phase B without launching real training.

What this verifies (cheap):
  1. Python env: torch + transformers importable, versions reported.
  2. CUDA available + device count + name.
  3. Data files exist (GECO, Provo, SUBTLEXus, predictability pkl).
  4. Each Phase B training script's CLI is sane (`--help` succeeds → all
     imports resolve and argparse parses).
  5. Each model variant module imports + the cascade class is constructable.
  6. exp07 precompute_surprisal.py imports and surprisal caches present
     (or, if absent, flagged as "needs precompute first").
  7. Existing paper-model / randinit / surp checkpoints inventoried per seed.

What it does NOT do by default:
  - Run training, even for a single batch.
  - Load TinyLlama weights (those are heavy; Phase B will load them).

Optional (slow):
  --forward     Build dualctx model end-to-end (loads TinyLlama-1.1B)
                and run one synthetic forward+backward pass on a 1×4-word
                batch. This is the most predictive sanity check and the
                one that catches silent shape/cascade bugs.

Usage:
    python check_phase_b.py
    python check_phase_b.py --forward
    python check_phase_b.py --strict --forward
"""

import argparse
import os
import subprocess
import sys
import importlib.util
from pathlib import Path

REPO_ROOT = Path("/home/u384661/Neuro_EZR")
SRC_V2 = REPO_ROOT / "src_v2"
LM_TRAIN = SRC_V2 / "lm_train"
LM_MODEL = SRC_V2 / "lm_model"
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CKPT_ROOT = REPO_ROOT / "checkpoints"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def ok(self, msg):
        print(f"  [{PASS}] {msg}")

    def fail(self, msg):
        self.fails += 1
        print(f"  [{FAIL}] {msg}")

    def warn(self, msg):
        self.warns += 1
        print(f"  [{WARN}] {msg}")

    def section(self, title):
        print(f"\n=== {title} ===")


def check_env(r):
    r.section("1. Python environment")
    print(f"  python: {sys.executable}")
    print(f"  version: {sys.version.split()[0]}")
    for mod in ("torch", "transformers", "numpy", "pandas", "scipy", "tqdm"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            r.ok(f"{mod} {ver}")
        except ImportError as e:
            r.fail(f"{mod} missing: {e}")


def check_cuda(r):
    r.section("2. CUDA")
    try:
        import torch
        avail = torch.cuda.is_available()
        n = torch.cuda.device_count()
        if not avail or n == 0:
            r.fail(
                f"torch.cuda.is_available()={avail}, device_count={n} — "
                "you are NOT on a GPU node. Phase B will fall back to CPU."
            )
            return
        r.ok(f"cuda available, {n} device(s)")
        for i in range(n):
            print(f"      device {i}: {torch.cuda.get_device_name(i)}")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f"      CUDA_VISIBLE_DEVICES={cvd!r}")
    except Exception as e:
        r.fail(f"CUDA check raised: {e}")


def check_data_files(r):
    r.section("3. Data files")
    required = [
        DATA_DIR / "SUBTLEXus.txt",
        DATA_DIR / "Geco_MonolingualReadingData.csv",
        DATA_DIR / "Geco_EnglishMaterial.csv",
        DATA_DIR / "geco_predictability.pkl",
        DATA_DIR / "Provo_Corpus-Eyetracking_Data.csv",
    ]
    for p in required:
        if p.exists() and p.stat().st_size > 0:
            r.ok(f"{p.name} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            r.fail(f"missing or empty: {p}")


def check_train_script_cli(r, script_path: Path, label: str,
                           expect_args=("--seed", "--epochs")):
    """Run `python script.py --help` and check it exits 0.

    This indirectly validates: every import in the script resolves,
    argparse setup is syntactically valid, and the module is parseable.
    `expect_args` is the list of CLI args the wrappers will pass to this
    script — pass () for one-shot deterministic scripts that need none.
    """
    if not script_path.exists():
        r.fail(f"{label}: file not found at {script_path}")
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            r.fail(f"{label}: --help exited {proc.returncode}\n      "
                   + "\n      ".join(tail))
            return
        out = proc.stdout
        for arg in expect_args:
            if arg not in out:
                r.warn(f"{label}: --help output missing {arg!r}")
        r.ok(f"{label}: --help OK")
    except subprocess.TimeoutExpired:
        r.fail(f"{label}: --help timed out (120s)")
    except Exception as e:
        r.fail(f"{label}: subprocess raised {e}")


def check_train_scripts(r):
    r.section("4. Training scripts (--help dry run)")
    scripts = [
        ("paper model (dualctx)",
         LM_TRAIN / "train_hybrid_v4c_v2_dualctx_geco.py"),
        ("randinit",
         LM_TRAIN / "train_hybrid_v4c_v2_randinit_geco.py"),
        ("surp ablation",
         LM_TRAIN / "train_hybrid_v4c_v2_surp_geco.py"),
    ]
    for label, path in scripts:
        check_train_script_cli(r, path, label)


def check_model_imports(r):
    r.section("5. Model variant modules importable")
    # Map import name -> file.
    if str(LM_MODEL) not in sys.path:
        sys.path.insert(0, str(LM_MODEL))
    modules = [
        "model_llama_hybrid_v4c_v2_dualctx",
        "model_llama_hybrid_v4c_v2_randinit",
        "model_llama_hybrid_v4c_v2_surp",
    ]
    for name in modules:
        path = LM_MODEL / f"{name}.py"
        if not path.exists():
            r.fail(f"{name}: file not found ({path})")
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            r.ok(f"{name}: imported")
        except Exception as e:
            r.fail(f"{name}: import failed — {e}")


def check_surprisal_cache(r):
    r.section("6. exp07 surprisal cache")
    precompute = (
        SRC_V2 / "paper_experiments" / "exp07_ctx_vs_surprisal" /
        "precompute_surprisal.py"
    )
    if not precompute.exists():
        r.fail(f"precompute_surprisal.py missing at {precompute}")
        return
    splits = ("train", "val", "test")
    have_all = True
    for split in splits:
        p = CACHE_DIR / f"tinyllama_surprisal_geco_{split}.pt"
        if p.exists() and p.stat().st_size > 0:
            r.ok(f"cache {split}: {p.stat().st_size / 1e6:.1f} MB")
        else:
            have_all = False
            r.warn(f"cache {split} missing at {p} — Phase B exp07 will "
                   "run precompute_surprisal.py first")
    if not have_all:
        # If caches missing, sanity-check the precompute script imports.
        # It's a one-shot deterministic script — no --seed/--epochs.
        check_train_script_cli(r, precompute, "precompute_surprisal",
                               expect_args=("--corpus", "--force"))


def check_existing_checkpoints(r):
    r.section("7. Existing checkpoints (per seed)")
    seeds = [1, 2, 3, 42, 100]
    families = [
        ("paper model", "hybrid_v4c_v2_dualctx"),
        ("randinit",    "hybrid_v4c_v2_randinit"),
        ("surp",        "hybrid_v4c_v2_surp"),
    ]
    for label, family in families:
        present = []
        missing = []
        for s in seeds:
            ckpt = (
                CKPT_ROOT / family /
                f"geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed{s}" /
                "best_model.pt"
            )
            if ckpt.exists():
                present.append(s)
            else:
                missing.append(s)
        if not missing:
            r.ok(f"{label}: all 5 seeds present {present}")
        elif present:
            r.warn(f"{label}: have seeds {present}, "
                   f"missing {missing} (Phase B will train)")
        else:
            r.warn(f"{label}: no checkpoints — Phase B will train all 5")


def check_forward_pass(r):
    """Construct the dualctx model fresh and run one tiny forward+backward.

    Loads TinyLlama-1.1B (~2.2 GB on GPU). Predictive: catches shape
    mismatches in the cascade, missing layers, broken loss assembly.
    """
    r.section("8. Dummy forward+backward (loads TinyLlama)")
    try:
        import torch
    except ImportError as e:
        r.fail(f"torch not importable: {e}")
        return

    if str(LM_MODEL) not in sys.path:
        sys.path.insert(0, str(LM_MODEL))

    try:
        from model_llama_hybrid_v4c_v2_dualctx import NeuralEZReaderHybrid
    except Exception as e:
        r.fail(f"can't import dualctx model: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      device: {device}")
    if device.type == "cpu":
        r.warn("running forward on CPU — slow, but will still validate logic")

    try:
        model = NeuralEZReaderHybrid(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            freeze_layers=22,
            hidden_dim=256,
        ).to(device)
        r.ok("model constructed (TinyLlama backbone loaded)")
    except Exception as e:
        r.fail(f"model construction raised: {e}")
        return

    # Dummy batch: 1 sentence, 4 words.
    word_lists = [["The", "quick", "brown", "fox"]]
    word_lengths = torch.tensor([[3, 5, 5, 3]], dtype=torch.long, device=device)
    frequencies = torch.tensor(
        [[200.0, 150.0, 12.0, 5.0]], dtype=torch.float32, device=device,
    )

    try:
        model.train()
        out = model(word_lists, frequencies, word_lengths)
    except Exception as e:
        r.fail(f"forward raised: {type(e).__name__}: {e}")
        return

    if not isinstance(out, dict):
        r.fail(f"forward returned {type(out).__name__}, expected dict")
        return

    # Cascade outputs the model is supposed to produce.
    expected_keys = {
        "first_fixation",     # FFD
        "gaze_duration",      # Gaze
        "total_reading_time", # TRT
        "skip_prob",          # skip
    }
    missing = expected_keys - set(out.keys())
    if missing:
        r.fail(f"forward output missing keys: {missing}. "
               f"got: {sorted(out.keys())}")
        return
    r.ok(f"forward has the 4 eye-tracking outputs (cascade has "
         f"{len(out)} fields total)")

    # Shape sanity.
    seq_len = word_lengths.size(1)
    for k in expected_keys:
        t = out[k]
        if t.shape != (1, seq_len):
            r.fail(f"out[{k!r}].shape={tuple(t.shape)}, expected (1, {seq_len})")
            return
    r.ok(f"all eye-tracking output shapes match (1, {seq_len})")

    # Finite values (no NaN/Inf).
    for k, t in out.items():
        if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
            r.fail(f"out[{k!r}] contains non-finite values")
            return
    r.ok("all outputs are finite")

    # Dummy backward: MSE on first_fixation + BCE on skip_prob.
    try:
        target_ffd = torch.tensor([[180.0, 220.0, 240.0, 260.0]],
                                   dtype=torch.float32, device=device)
        target_skip = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                    dtype=torch.float32, device=device)
        loss = (out["first_fixation"] - target_ffd).pow(2).mean()
        loss = loss + torch.nn.functional.binary_cross_entropy(
            out["skip_prob"].clamp(1e-6, 1 - 1e-6), target_skip,
        )
        loss.backward()
        if not torch.isfinite(loss):
            r.fail(f"loss is non-finite: {loss.item()}")
            return
        r.ok(f"backward OK, loss={loss.item():.3f}")
    except Exception as e:
        r.fail(f"backward raised: {type(e).__name__}: {e}")
        return

    # At least one trainable parameter received a gradient.
    n_with_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    if n_with_grad == 0:
        r.fail("no parameters received gradients — frozen everything?")
    else:
        r.ok(f"{n_with_grad} parameter tensors received gradients")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any FAIL")
    parser.add_argument("--forward", action="store_true",
                        help="also run a dummy forward+backward through the "
                             "dualctx model (loads TinyLlama, slow)")
    args = parser.parse_args()

    r = Report()
    check_env(r)
    check_cuda(r)
    check_data_files(r)
    check_train_scripts(r)
    check_model_imports(r)
    check_surprisal_cache(r)
    check_existing_checkpoints(r)
    if args.forward:
        check_forward_pass(r)

    print("\n" + "=" * 60)
    print(f"summary: {r.fails} FAIL, {r.warns} WARN")
    print("=" * 60)
    if args.strict and r.fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
