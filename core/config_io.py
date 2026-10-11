"""Save / load the E-PROM configuration (criteria setup and run options)."""

from __future__ import annotations
import json
from datetime import datetime

from .labels import match_key

CONFIG_VERSION = 1
PREFERENCE_NAMES = ("Usual", "U-shape", "V-shape", "Level", "Linear")
DIRECTIONS = ("Maximize", "Minimize")
TYPES = ("Ordinal", "Continuous")


def build_config(
    criteria, types, directions, preferences, q_vals, p_vals, scales,
    mc_variation=None, expert_weights=None, expert_names=None, monte_carlo=None,
):
    """Assemble a JSON-serializable configuration dictionary.

    ``directions`` use +1 / -1 (as in the core); ``preferences`` are names.
    """
    crit = []
    for k, name in enumerate(criteria):
        crit.append({
            "name": str(name),
            "type": str(types[k]),
            "direction": "Maximize" if directions[k] == 1 else "Minimize",
            "preference": str(preferences[k]),
            "q": float(q_vals[k]),
            "p": float(p_vals[k]),
            "scale": int(scales[k]) if str(types[k]) == "Ordinal" else None,
            "mc_variation_pct": (
                float(mc_variation[k]) if mc_variation is not None else None
            ),
        })

    cfg = {
        "app": "E-PROM",
        "config_version": CONFIG_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "criteria": crit,
    }
    if expert_names is not None and expert_weights is not None:
        cfg["decision_makers"] = {
            str(n): float(w) for n, w in zip(expert_names, expert_weights)
        }
    if monte_carlo:
        cfg["monte_carlo"] = dict(monte_carlo)
    return cfg


def dumps(config):
    return json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8")


def parse_config(raw):
    """Parse and minimally validate a configuration file (bytes or str)."""
    try:
        cfg = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception as exc:
        raise ValueError(f"The file is not valid JSON: {exc}") from None

    if not isinstance(cfg, dict) or cfg.get("app") != "E-PROM" or "criteria" not in cfg:
        raise ValueError("This does not look like an E-PROM configuration file.")
    if not isinstance(cfg["criteria"], list):
        raise ValueError("Invalid configuration: 'criteria' must be a list.")
    return cfg


def criteria_defaults(cfg, criteria):
    """Match the saved criteria to the current ones (by name).

    Returns ({criterion index: settings}, warnings).
    """
    saved = {match_key(c.get("name")): c for c in cfg["criteria"]}
    out, warnings = {}, []

    for k, name in enumerate(criteria):
        entry = saved.get(match_key(name))
        if entry is None:
            warnings.append(f"Criterion '{name}' is not in the configuration file; defaults kept.")
            continue

        typ = entry.get("type")
        pref = entry.get("preference")
        direction = entry.get("direction")
        if typ not in TYPES or pref not in PREFERENCE_NAMES or direction not in DIRECTIONS:
            warnings.append(f"Criterion '{name}': invalid saved settings; defaults kept.")
            continue

        out[k] = {
            "type": typ,
            "direction": direction,
            "preference": pref,
            "q": float(entry.get("q") or 0.0),
            "p": float(entry.get("p") or 0.0),
            "scale": entry.get("scale") if entry.get("scale") in (5, 7, 9) else 7,
            "mc_variation_pct": entry.get("mc_variation_pct"),
        }

    extra = [c.get("name") for c in cfg["criteria"] if match_key(c.get("name")) not in {match_key(x) for x in criteria}]
    if extra:
        warnings.append("Saved criteria not in the current workbook were ignored: " + ", ".join(map(str, extra)))
    return out, warnings


def expert_weight_defaults(cfg, expert_names):
    saved = {match_key(k): v for k, v in (cfg.get("decision_makers") or {}).items()}
    return [float(saved.get(match_key(n), 1.0)) for n in expert_names]
