#!/usr/bin/env python3
"""Automated experiment runner for ProCyon.

Reads config.yaml, tracks state in state.json, runs experiments in order.
Usage:
  python experiments/runner.py              # run next pending experiment
  python experiments/runner.py --list       # show all experiments and status
  python experiments/runner.py --phase 1    # run all experiments in phase 1
  python experiments/runner.py --dry-run    # show what would run next
  python experiments/runner.py --retry H1.1 # re-run a specific experiment
  python experiments/runner.py --all        # run everything pending
"""
import argparse, json, os, subprocess, sys, time, yaml
from datetime import datetime
from pathlib import Path

PROJECT = "/hd/liujx/microbiome_llm_project"
CONFIG = os.path.join(PROJECT, "experiments/config.yaml")
STATE = os.path.join(PROJECT, "experiments/state.json")
RESULT_DIR = os.path.join(PROJECT, "experiments/results")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)

def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}

def save_state(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def load_results(exp_id):
    path = os.path.join(RESULT_DIR, f"{exp_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_results(exp_id, results):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{exp_id}.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def check_deps_done(exp_id, config, state):
    """Check that all dependencies are done."""
    exp = config['experiments'][exp_id]
    for dep in exp.get('depends', []):
        dep_status = state.get(dep, {}).get('status', STATUS_PENDING)
        if dep_status != STATUS_DONE:
            return False, dep
    return True, None

def next_pending(config, state, phase=None):
    """Find the next experiment to run, respecting phase order."""
    experiments = config['experiments']
    phase_order = config.get('phase_order', [1, 2, 3, 4])

    candidates = []
    for exp_id, exp in experiments.items():
        current = state.get(exp_id, {}).get('status', STATUS_PENDING)
        if current != STATUS_PENDING:
            continue
        if phase is not None and exp['phase'] != phase:
            continue
        deps_ok, _ = check_deps_done(exp_id, config, state)
        if not deps_ok:
            continue
        candidates.append((exp['phase'], phase_order.index(exp['phase']) if exp['phase'] in phase_order else 99, exp_id, exp))

    candidates.sort(key=lambda x: (x[1], x[0], x[2]))
    return candidates[0][2] if candidates else None

def run_experiment(exp_id, config, state, dry_run=False):
    """Run a single experiment."""
    exp = config['experiments'][exp_id]
    script = os.path.join(PROJECT, exp['script'])

    if not os.path.exists(script):
        print(f"[SKIP] {exp_id}: script not found ({script})")
        state[exp_id] = {'status': STATUS_SKIPPED, 'reason': 'script not found', 'time': str(datetime.now())}
        save_state(state)
        return

    print(f"\n{'='*60}")
    print(f"[RUN] {exp_id}: {exp['desc']}")
    print(f"  Script: {script}")
    print(f"  Hypothesis: {exp.get('hypothesis', 'N/A')}")
    print(f"  Timeout: {exp.get('timeout_min', 30)} min")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY-RUN] Would execute but --dry-run is set.")
        return

    # Mark running
    state[exp_id] = {'status': STATUS_RUNNING, 'started': str(datetime.now()), 'script': script}
    save_state(state)

    gpu = str(exp.get('gpu', 1))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu

    try:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT,
            env=env,
            capture_output=True,
            text=True,
            timeout=exp.get('timeout_min', 30) * 60
        )
        elapsed = time.time() - t0

        # Collect output
        log_path = os.path.join(PROJECT, f"experiments/logs/{exp_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(proc.stderr)

        if proc.returncode == 0:
            # Try to load structured results
            result_path = os.path.join(RESULT_DIR, f"{exp_id}.json")
            if os.path.exists(result_path):
                results = json.load(open(result_path))
            else:
                results = {'stdout_tail': proc.stdout[-2000:] if proc.stdout else ''}

            results['runtime_min'] = round(elapsed / 60, 1)
            results['returncode'] = 0
            save_results(exp_id, results)

            state[exp_id].update({
                'status': STATUS_DONE,
                'completed': str(datetime.now()),
                'runtime_min': round(elapsed / 60, 1)
            })
            print(f"[DONE] {exp_id}: completed in {elapsed/60:.1f}min")
            # Print key metrics if available
            if 'metrics' in results:
                for k, v in results['metrics'].items():
                    print(f"  {k}: {v}")
        else:
            state[exp_id].update({
                'status': STATUS_FAILED,
                'completed': str(datetime.now()),
                'returncode': proc.returncode,
                'stderr_tail': (proc.stderr or '')[-1000:]
            })
            print(f"[FAIL] {exp_id}: returncode={proc.returncode}")
            print(f"  stderr: {(proc.stderr or '')[-500:]}")

    except subprocess.TimeoutExpired:
        state[exp_id].update({
            'status': STATUS_FAILED,
            'completed': str(datetime.now()),
            'reason': 'timeout'
        })
        print(f"[TIMEOUT] {exp_id}: exceeded {exp.get('timeout_min', 30)}min")
    except Exception as e:
        state[exp_id].update({
            'status': STATUS_FAILED,
            'completed': str(datetime.now()),
            'reason': str(e)
        })
        print(f"[ERROR] {exp_id}: {e}")

    save_state(state)

def list_experiments(config, state):
    """Print all experiments with status."""
    print(f"\n{'ID':<8} {'Phase':<6} {'Status':<10} {'Depends':<20} {'Desc'}")
    print("-" * 90)
    for exp_id, exp in config['experiments'].items():
        st = state.get(exp_id, {}).get('status', STATUS_PENDING)
        icon = {'done': '🟢', 'running': '🔄', 'failed': '🔴', 'pending': '⬜', 'skipped': '⏭️'}.get(st, '⬜')
        deps = ', '.join(exp.get('depends', [])) or '-'
        print(f"{icon} {exp_id:<6} P{exp['phase']:<5} {st:<10} {deps:<20} {exp['desc']}")

def sync_state_from_results(config, state):
    """Sync state from result files (recover after restart)."""
    changed = False
    for exp_id in config['experiments']:
        results = load_results(exp_id)
        if results and state.get(exp_id, {}).get('status') != STATUS_DONE:
            state[exp_id] = {
                'status': STATUS_DONE,
                'completed': results.get('timestamp', str(datetime.now())),
                'runtime_min': results.get('runtime_min', 0)
            }
            changed = True
    if changed:
        save_state(state)
        print("[SYNC] Updated state from existing result files.")
    return state

def main():
    parser = argparse.ArgumentParser(description="ProCyon Experiment Runner")
    parser.add_argument('--list', action='store_true', help='List all experiments and status')
    parser.add_argument('--phase', type=int, help='Run all pending experiments in phase N')
    parser.add_argument('--all', action='store_true', help='Run all pending experiments')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run without executing')
    parser.add_argument('--retry', type=str, help='Re-run a specific experiment (even if done)')
    args = parser.parse_args()

    config = load_config()
    state = load_state()
    state = sync_state_from_results(config, state)

    if args.list:
        list_experiments(config, state)
        return

    if args.retry:
        exp_id = args.retry
        if exp_id not in config['experiments']:
            print(f"Unknown experiment: {exp_id}")
            return
        # Reset status to pending
        state.pop(exp_id, None)
        save_state(state)
        run_experiment(exp_id, config, state, dry_run=args.dry_run)
        return

    if args.phase:
        print(f"Running all pending experiments in Phase {args.phase}...")
        while True:
            exp_id = next_pending(config, state, phase=args.phase)
            if exp_id is None:
                print(f"\nPhase {args.phase} complete - no more pending experiments.")
                break
            run_experiment(exp_id, config, state, dry_run=args.dry_run)
            if args.dry_run:
                break
            # Small gap between experiments
            time.sleep(5)
        return

    if args.all:
        print("Running all pending experiments...")
        while True:
            exp_id = next_pending(config, state)
            if exp_id is None:
                print("\nAll experiments complete!")
                break
            run_experiment(exp_id, config, state, dry_run=args.dry_run)
            if args.dry_run:
                break
            time.sleep(5)
        return

    # Default: run next single experiment
    exp_id = next_pending(config, state)
    if exp_id is None:
        print("No pending experiments (all done or blocked by dependencies).")
        list_experiments(config, state)
        return
    run_experiment(exp_id, config, state, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
