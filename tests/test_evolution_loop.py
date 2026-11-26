"""Run the complete generation/screen/replay loop using deterministic local model responses."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillopt_evolution import evolve, run, usage


class LocalModels:
    def __init__(self):
        self.optimizer_calls = 0
        self.evaluated_items = 0
        self.inputs = 0
        self.outputs = 0
        self.events = []

    def counters(self):
        return {"_total": {"calls": self.optimizer_calls + self.evaluated_items,
                           "prompt_tokens": self.inputs, "completion_tokens": self.outputs,
                           "total_tokens": self.inputs + self.outputs}}

    def optimize(self, *, user, **kwargs):
        self.optimizer_calls += 1
        self.inputs += 12
        self.outputs += 3
        self.events.append("generate")
        context = json.loads(user)
        sources = [
            {"donor": role, "field": "skill", "edit_index": None,
             "quote": context["documents"][context["role_to_document"][role]][:24]}
            for role in ("A", "B")
        ]
        proposal = {
            "sources": sources, "difference": "Use evidence from different documents.",
            "mutation": "Narrow the cases where the rule applies.",
            "best_integration": "Preserve the best rule's scope.", "crossover": "Append the narrowed rule.",
            "hoped_for": "Fix a second type of question.", "at_risk": "Unnecessary qualifications.",
            "edits": [{"op": "append", "content": f"Offspring rule {self.optimizer_calls}."}],
        }
        return json.dumps(proposal), {"prompt_tokens": 12, "completion_tokens": 3}

    def rollout(self, items, skill, directory):
        self.events.append("validate")
        self.evaluated_items += len(items)
        self.inputs += 8 * len(items)
        self.outputs += 2 * len(items)
        # Only slot 2 of generation 1 improves: 4/8 vs parent 3/8, below best 6/8.
        number_correct = 4 if skill.rstrip().endswith("Offspring rule 2.") else 0
        return [dict(item, agent_ok=True, hard=int(int(item["id"]) < number_correct)) for item in items]


class EvolutionLoopTests(unittest.TestCase):
    def test_frozen_generation_local_replacement_patience_and_cost_free_resume(self):
        items = [{"id": str(i), "question": f"Development question {i}?"} for i in range(8)]
        population = []
        for uid, count in [("strong", 6), ("weak", 3)]:
            text = "Read all evidence before responding, " + uid + "."
            population.append({
                "id": uid, "skill": text, "sha256": run.sha(text), "correct": count,
                "score": count / len(items), "complete": True, "edits": [], "action": "accept_new_best",
                "parent_sha256": None, "rows": {it["id"]: dict(it, agent_ok=True, hard=int(i < count))
                                               for i, it in enumerate(items)},
            })
        archive = {
            "population": population, "records": population, "skipped": [],
            "native_step_usage": {"input_tokens": 17, "output_tokens": 3, "total_tokens": 20},
            "initial_population": {"selected_count": 2, "population_cap": 5, "shortfall": 3},
            "history": {it["id"]: [p["rows"][it["id"]]["hard"] for p in population] for it in items},
        }
        models = LocalModels()
        options = {"seed": 42, "strategy": "random", "edit_budget": 2, "generations": 7, "patience": 2}
        with tempfile.TemporaryDirectory() as folder, patch.object(
            evolve, "chat_optimizer", side_effect=models.optimize
        ), patch.object(usage, "get_token_summary", side_effect=models.counters):
            out = Path(folder)
            kwargs = dict(adapter=models, archive=archive, out=out, items=items, options=options)
            summary = run.run_loop(**kwargs)
            generations = summary["generations"]
            self.assertEqual(summary["stop_reason"], "consecutive_generations_without_replacement")
            self.assertEqual([g["replacements"] for g in generations], [1, 0, 0])
            self.assertEqual([g["flat_generations"] for g in generations], [0, 1, 2])
            self.assertEqual(generations[0]["population"][1]["correct"], 4)
            self.assertEqual(summary["best"]["id"], "strong")
            next_plan = usage.read_json(out / "generations/002/frozen.json")
            self.assertEqual(next_plan["parents"][1]["id"], "g001s02")
            for generation in range(1, 4):
                frozen = usage.read_json(out / f"generations/{generation:03d}/frozen.json")
                for slot in (1, 2):
                    inputs = usage.read_json(out / f"generations/{generation:03d}/slot_{slot:02d}/evaluation_inputs.json")
                    self.assertEqual(frozen["order"], inputs["signature"]["frozen_order"])
            # The optimizer never sees the same generation's evaluation.
            blocks = [event for event, prev in zip(models.events, [None, *models.events]) if event != prev]
            self.assertEqual(blocks, ["generate", "validate"] * 3)
            self.assertEqual(models.optimizer_calls, 6)
            costs = summary["incremental_usage"]
            self.assertEqual(costs["generation"]["input_tokens"], 6 * 12)
            self.assertEqual(costs["generation"]["output_tokens"], 6 * 3)
            self.assertEqual(costs["validation"]["input_tokens"], models.evaluated_items * 8)
            self.assertEqual(costs["validation"]["output_tokens"], models.evaluated_items * 2)
            self.assertEqual(costs["total"]["total_tokens"], models.inputs + models.outputs)
            before = list(models.events)
            self.assertEqual(run.run_loop(**kwargs), summary)
            self.assertEqual(models.events, before)
