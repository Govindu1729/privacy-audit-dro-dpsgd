"""
Smart Checkpoint Manager
=========================
Resume, quality assessment, archiving, decision engine.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn


class RunDecision(Enum):
    RESUME_TRAINING = "resume_training"
    RESUME_AUDIT = "resume_audit"
    START_FRESH = "start_fresh"
    COMPLETE = "complete"


class CheckpointManager:
    """Manages experiment checkpoints with intelligent resume decisions."""
    
    def __init__(self, exp_dir, experiment_name="experiment"):
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metadata_file = self.ckpt_dir / "metadata.json"
        self.decision_log_file = self.ckpt_dir / "decision_log.json"
        self.config_hash_file = self.ckpt_dir / "config_hash.txt"
        self.model_hash_file = self.ckpt_dir / "model_hash.txt"
        
        self.decision_log = []
        self._load_decision_log()
    
    # ── Hashing ──────────────────────────────────────────────────────────
    
    def compute_config_hash(self, config_dict):
        raw = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def compute_model_hash(self, model):
        base = model._module if hasattr(model, '_module') else model
        arch_str = ""
        for name, param in base.named_parameters():
            arch_str += f"{name}:{list(param.shape)};"
        return hashlib.sha256(arch_str.encode()).hexdigest()[:16]
    
    def save_hashes(self, config_dict, model):
        self.config_hash_file.write_text(self.compute_config_hash(config_dict))
        self.model_hash_file.write_text(self.compute_model_hash(model))
    
    def load_previous_hashes(self):
        cfg_hash = self.config_hash_file.read_text().strip() if self.config_hash_file.exists() else None
        model_hash = self.model_hash_file.read_text().strip() if self.model_hash_file.exists() else None
        return cfg_hash, model_hash
    
    # ── Metadata ─────────────────────────────────────────────────────────
    
    def save_metadata(self, metadata):
        metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_decision_log(self):
        if self.decision_log_file.exists():
            with open(self.decision_log_file, 'r') as f:
                self.decision_log = json.load(f)
    
    def log_decision(self, decision, reason, details=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision.value,
            'reason': reason,
            'details': details or {}
        }
        self.decision_log.append(entry)
        with open(self.decision_log_file, 'w') as f:
            json.dump(self.decision_log, f, indent=2)
    
    # ── Training Checkpoints ─────────────────────────────────────────────
    
    def save_training_checkpoint(self, phase, epoch, model, optimizer,
                                  history, dro_state=None, extra=None):
        phase_dir = self.ckpt_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        base_model = model._module if hasattr(model, '_module') else model
        state = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'dro_state': dro_state or {},
            'extra': extra or {},
            'timestamp': datetime.now().isoformat()
        }
        
        ckpt_path = phase_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, ckpt_path)
        torch.save(state, phase_dir / "latest.pt")
        return ckpt_path
    
    def load_training_checkpoint(self, phase, epoch=None):
        phase_dir = self.ckpt_dir / phase
        if epoch is not None:
            path = phase_dir / f"epoch_{epoch:03d}.pt"
        else:
            path = phase_dir / "latest.pt"
        
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=False)
        return None
    
    # ── Data Split ───────────────────────────────────────────────────────
    
    def save_data_split(self, split_data):
        torch.save(split_data, self.ckpt_dir / "data_split.pt")
    
    def load_data_split(self):
        path = self.ckpt_dir / "data_split.pt"
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=False)
        return None
    
    # ── Decision Engine ──────────────────────────────────────────────────
    
    def decide_action(self, config_dict, model):
        metadata = self.load_metadata()
        prev_cfg_hash, prev_model_hash = self.load_previous_hashes()
        curr_cfg_hash = self.compute_config_hash(config_dict)
        curr_model_hash = self.compute_model_hash(model)
        
        if metadata is None:
            reason = "No previous run found. Starting fresh."
            self.log_decision(RunDecision.START_FRESH, reason)
            return RunDecision.START_FRESH, reason
        
        if prev_model_hash and prev_model_hash != curr_model_hash:
            reason = f"Model architecture changed ({prev_model_hash} -> {curr_model_hash}). Starting fresh."
            self.log_decision(RunDecision.START_FRESH, reason)
            return RunDecision.START_FRESH, reason
        
        if prev_cfg_hash and prev_cfg_hash != curr_cfg_hash:
            reason = f"Config changed ({prev_cfg_hash} -> {curr_cfg_hash}). Starting fresh."
            self.log_decision(RunDecision.START_FRESH, reason)
            return RunDecision.START_FRESH, reason
        
        if metadata.get('status') == 'complete':
            reason = "Experiment already complete. Delete checkpoints to re-run."
            self.log_decision(RunDecision.COMPLETE, reason)
            return RunDecision.COMPLETE, reason
        
        completed = metadata.get('completed_phases', [])
        current_stage = metadata.get('current_stage', 'none')
        
        if 'audit' in completed:
            reason = "Audit complete. Loading results."
            self.log_decision(RunDecision.COMPLETE, reason)
            return RunDecision.COMPLETE, reason
        
        if 'dro_training' in completed:
            reason = "DRO training complete. Resuming from audit."
            self.log_decision(RunDecision.RESUME_AUDIT, reason)
            return RunDecision.RESUME_AUDIT, reason
        
        if 'erm_training' in completed:
            reason = "ERM complete. Resuming DRO training."
            self.log_decision(RunDecision.RESUME_TRAINING, reason)
            return RunDecision.RESUME_TRAINING, reason
        
        if current_stage in ['erm_training', 'dro_training']:
            phase = 'erm' if 'erm' in current_stage else 'dro'
            ckpt = self.load_training_checkpoint(phase)
            if ckpt:
                reason = f"Resuming {phase.upper()} from epoch {ckpt['epoch']}."
                self.log_decision(RunDecision.RESUME_TRAINING, reason)
                return RunDecision.RESUME_TRAINING, reason
        
        reason = "Could not determine resume point. Starting fresh."
        self.log_decision(RunDecision.START_FRESH, reason)
        return RunDecision.START_FRESH, reason
    
    # ── Archiving ────────────────────────────────────────────────────────
    
    def archive_previous_run(self):
        if self.ckpt_dir.exists() and any(self.ckpt_dir.iterdir()):
            archive_name = f"checkpoints_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive_path = self.exp_dir / archive_name
            shutil.move(str(self.ckpt_dir), str(archive_path))
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            return archive_path
        return None
    
    def print_decision(self, decision, reason):
        print("\n" + "=" * 78)
        icons = {
            RunDecision.START_FRESH: "DECISION: STARTING FRESH RUN",
            RunDecision.RESUME_TRAINING: "DECISION: RESUMING TRAINING",
            RunDecision.RESUME_AUDIT: "DECISION: RESUMING FROM AUDIT",
            RunDecision.COMPLETE: "DECISION: EXPERIMENT COMPLETE",
        }
        print(f"  {icons.get(decision, 'DECISION')}")
        print("=" * 78)
        print(f"  Reason: {reason}")
        print("=" * 78 + "\n")
    
    # ── Results ──────────────────────────────────────────────────────────
    
    def save_final_results(self, results):
        final_path = self.ckpt_dir / "final_results.json"
        with open(final_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_final_results(self):
        final_path = self.ckpt_dir / "final_results.json"
        if final_path.exists():
            with open(final_path, 'r') as f:
                return json.load(f)
        return None
