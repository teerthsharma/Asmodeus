import logging
from typing import List, Dict, Tuple, Optional

from asmodeus.registry import Registry, MicroMaster

logger = logging.getLogger(__name__)

class SparseTopKRouter:
    """
    Implements S_t = topk(r^t, k) routing rule for the micro-master swarm.
    Incorporates model-cost-aware gating, reliability, utilization telemetry,
    and parameter budget constraints.
    """
    def __init__(self, registry: Registry, k: int = 5, max_active_budget: Optional[int] = None):
        self.registry = registry
        self.k = k
        self.max_active_budget = max_active_budget
        self.utilization_telemetry: Dict[str, int] = {}
        
        # Hyperparameters for routing score equation:
        # r_i^t = beta_1 * sim(q_t, c_i) + beta_2 * rel_i - beta_3 * cost_i + beta_4 * fresh_i
        self.beta_1 = 1.0  # Capability similarity weight
        self.beta_2 = 0.5  # Reliability weight
        self.beta_3 = 0.2  # Cost penalty weight
        self.beta_4 = 0.1  # Utilization balance (entropy/freshness) weight

    def _compute_similarity(self, task_tags: List[str], model: MicroMaster) -> float:
        """
        Computes sim(q_t, c_i).
        Mock similarity based on Jaccard index of capability tags.
        Normalizes tag strings (lowercase, stripped) for consistent matching.
        """
        if not task_tags or not model.capability_tags:
            return 0.0

        norm_task_tags = {str(tag).strip().lower() for tag in task_tags if str(tag).strip()}
        norm_model_tags = {str(tag).strip().lower() for tag in model.capability_tags if str(tag).strip()}

        if not norm_task_tags or not norm_model_tags:
            return 0.0

        intersection = len(norm_task_tags.intersection(norm_model_tags))
        union = len(norm_task_tags.union(norm_model_tags))
        return intersection / union if union > 0 else 0.0

    def route(
        self,
        task_tags: List[str],
        k_override: Optional[int] = None,
        max_active_budget: Optional[int] = None,
    ) -> List[MicroMaster]:
        """
        Routes a task to the top-k specialists.
        Active set: S_t = topk(r^t, k)
        Constraint: S_t subseteq M_ready(t)
        Optional budget constraint: sum(P_i) <= max_active_budget
        """
        k = k_override if k_override is not None else self.k
        budget = max_active_budget if max_active_budget is not None else self.max_active_budget

        ready_models = self.registry.get_ready_models()
        
        if not ready_models:
            logger.warning("No ready models available for routing in M_ready(t).")
            return []

        scored_models: List[Tuple[float, MicroMaster]] = []
        
        for model in ready_models:
            # 1. Similarity score: sim(q_t, c_i)
            sim_score = self._compute_similarity(task_tags, model)
            
            # 2. Historical reliability: rel_i
            rel_score = model.reliability
            
            # 3. Cost metric: cost_i
            cost_penalty = model.cost_per_call
            
            # 4. Utilization entropy / load balance: fresh_i equivalent
            # Penalize heavily used models to avoid specialist collapse
            usage_count = self.utilization_telemetry.get(model.model_id, 0)
            balance_score = 1.0 / (1.0 + usage_count) 

            # Routing score r_i^t
            r_t = (self.beta_1 * sim_score) + \
                  (self.beta_2 * rel_score) - \
                  (self.beta_3 * cost_penalty) + \
                  (self.beta_4 * balance_score)

            # Unavailable/unverified models are already excluded by get_ready_models()

            scored_models.append((r_t, model))

        # Sort descending by score r_t
        scored_models.sort(key=lambda x: x[0], reverse=True)

        top_candidates = [model for score, model in scored_models[:k]]

        # Apply active parameter budget guardrail if configured
        if budget is not None:
            top_k_models = []
            accumulated_params = 0
            for model in top_candidates:
                if accumulated_params + model.parameter_count <= budget:
                    top_k_models.append(model)
                    accumulated_params += model.parameter_count
                elif not top_k_models and model.parameter_count > budget:
                    logger.warning(
                        f"Model {model.model_id} parameters ({model.parameter_count}) "
                        f"exceed max budget ({budget})."
                    )
        else:
            top_k_models = top_candidates

        # Update utilization telemetry
        for model in top_k_models:
            self.utilization_telemetry[model.model_id] = self.utilization_telemetry.get(model.model_id, 0) + 1

        return top_k_models

    def get_telemetry(self) -> Dict[str, int]:
        """
        Returns the utilization telemetry.
        Can be used to monitor specialist collapse rate and utilization entropy.
        """
        return self.utilization_telemetry

    def get_active_parameters(self, models: List[MicroMaster]) -> int:
        """
        Calculates the aggregate active parameter footprint of the selected models.
        """
        return sum(m.parameter_count for m in models)
