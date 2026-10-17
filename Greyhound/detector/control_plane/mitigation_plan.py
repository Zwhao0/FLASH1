import math
import logging
import time
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy
from .global_analyzer import PerformanceMetric, CommunicatorClique
from .dp_planner import get_time_array, solve_dp
from .pp_planner import apply_layer_split
from .cost_estimator import CostEstimator


class MitigationPlan(object):
    def __init__(self, redis_client):
        self.client = redis_client
        self.dp_version = 0
        self.estimator = CostEstimator()
        self.rs_cost = self.estimator.get_restart_adjustment_cost()

    def group_comp_time_by_dp(self,
                              all_cliques: Dict[int, CommunicatorClique],
                              comp_results: Dict[int, PerformanceMetric]):
        comp_results = deepcopy(comp_results)
        # 1st pass: Synchronize all results within the same PP group
        for c in all_cliques.values():
            if c.is_pp:
                ts = [comp_results[i].avg_lat for i in c.ranks]
                max_ts = max(ts)
                for r in c.ranks:
                    comp_results[r] = PerformanceMetric(
                        max_ts, max_ts, max_ts, 0
                    )
        # 2nd pass: Synchronize all results within the same TP group
        for c in all_cliques.values():
            if c.is_tp:
                ts = [comp_results[i].avg_lat for i in c.ranks]
                max_ts = max(ts)
                for r in c.ranks:
                    comp_results[r] = PerformanceMetric(
                        max_ts, max_ts, max_ts, 0
                    )
        # 3rd pass: aggregate to DP group
        dp_results = {}
        for c in all_cliques.values():
            if c.is_dp:
                for (dpr, r) in enumerate(c.ranks):
                    dp_results[dpr] = comp_results[r]
        logging.info(f"DP computation results: {dp_results}")
        return dp_results

    def parse_communication_results(
            self,
            comm_results: Dict[Tuple[int], List],
            comm_cliques: Dict[int, CommunicatorClique]):
        """Return worst observed P2P latency for each actual PP rank.

        ``comm_results`` can contain results from more than one communicator
        type. The old implementation sorted all communicator keys and grouped
        them by TP degree. That accidentally mixed PP, TP, and DP cliques, and
        its sliding window could index beyond the last entry. PP stage identity
        comes from Megatron's per-rank ``*_pp`` rank list, whose order is PP
        rank order.
        """
        pp_cliques = {
            tuple(sorted(clique.ranks))
            for clique in comm_cliques.values()
            if clique.is_pp
        }
        if not pp_cliques:
            logging.warning("No pipeline-parallel communicator cliques are available")
            return []

        stage_count = None
        stage_latency = {}
        for communicator_ranks, task_results in comm_results.items():
            clique_ranks = tuple(sorted(int(rank) for rank in communicator_ranks))
            if clique_ranks not in pp_cliques:
                continue

            stage_by_rank = {}
            clique_pp_order = None
            for rank in clique_ranks:
                raw_pp_ranks = self.client.get(f"{rank}_pp")
                if raw_pp_ranks is None:
                    raise ValueError(f"Missing Redis pipeline-rank list for global rank {rank}")
                if isinstance(raw_pp_ranks, bytes):
                    raw_pp_ranks = raw_pp_ranks.decode()
                pp_ranks = tuple(int(value) for value in raw_pp_ranks.split("_"))
                if len(pp_ranks) != len(clique_ranks) or set(pp_ranks) != set(clique_ranks):
                    raise ValueError(
                        f"Rank {rank} pipeline group {pp_ranks} does not match "
                        f"the measured PP clique {clique_ranks}"
                    )
                if clique_pp_order is None:
                    clique_pp_order = pp_ranks
                elif clique_pp_order != pp_ranks:
                    raise ValueError(
                        f"Ranks in PP clique {clique_ranks} disagree on PP rank order: "
                        f"{clique_pp_order} vs {pp_ranks}"
                    )
                if rank not in pp_ranks:
                    raise ValueError(f"Global rank {rank} is missing from its PP group {pp_ranks}")
                if stage_count is None:
                    stage_count = len(pp_ranks)
                elif stage_count != len(pp_ranks):
                    raise ValueError(
                        f"Inconsistent pipeline sizes: {stage_count} and {len(pp_ranks)}"
                    )
                stage_by_rank[rank] = pp_ranks.index(rank)

            for task_result in task_results:
                for rank, stage in stage_by_rank.items():
                    metric = task_result.get(rank)
                    if metric is None:
                        continue
                    latency = float(metric.avg_lat)
                    if not math.isfinite(latency) or latency <= 0:
                        continue
                    stage_latency[stage] = max(stage_latency.get(stage, 0.0), latency)

        if stage_count is None:
            logging.warning("Communication results contain no target PP clique measurements")
            return []
        missing_stages = [stage for stage in range(stage_count) if stage not in stage_latency]
        if missing_stages:
            raise ValueError(f"No valid communication measurements for PP stages {missing_stages}")

        max_comm_time_per_pp_stage = [stage_latency[stage] for stage in range(stage_count)]
        logging.info(f"Max communication time by PP stage: {max_comm_time_per_pp_stage}")
        return max_comm_time_per_pp_stage

    @staticmethod
    def split_layers_by_stage_latency(total_num_layers: int, stage_latencies: List[float]):
        """Assign every transformer layer to a stage, favoring faster links."""
        pp_size = len(stage_latencies)
        if pp_size < 2:
            raise ValueError(f"Pipeline repartition requires at least two stages, got {pp_size}")
        if total_num_layers < pp_size:
            raise ValueError(
                f"Cannot assign positive layer counts to {pp_size} stages from "
                f"only {total_num_layers} layers"
            )

        latencies = [float(value) for value in stage_latencies]
        if any(not math.isfinite(value) or value <= 0 for value in latencies):
            raise ValueError(f"PP stage latencies must be finite and positive: {latencies}")
        weights = [1.0 / value for value in latencies]
        weight_sum = sum(weights)

        # Reserve one layer for every stage, then use the largest-remainder
        # method so integer rounding cannot create empty stages or lose layers.
        distributable = total_num_layers - pp_size
        exact = [distributable * weight / weight_sum for weight in weights]
        extra = [math.floor(value) for value in exact]
        remainder = distributable - sum(extra)
        order = sorted(range(pp_size), key=lambda index: (exact[index] - extra[index], -index), reverse=True)
        for index in order[:remainder]:
            extra[index] += 1
        layer_split = [1 + value for value in extra]
        if len(layer_split) != pp_size or sum(layer_split) != total_num_layers or min(layer_split) < 1:
            raise AssertionError(f"Invalid PP layer split {layer_split} for {total_num_layers} layers")
        return layer_split

    def adjust_batchsize_distribution(self,
                                      comp_results: Dict[int, PerformanceMetric],
                                      cliques: Dict[int, CommunicatorClique]):
        comp_results_by_dp = self.group_comp_time_by_dp(cliques, comp_results)
        time_array = get_time_array(self.client, comp_results_by_dp)
        # Mitigation by adjusting batch size distribution across DP groups
        micro_bsz = self.client.get("micro_batch_size")
        global_bsz = self.client.get("global_batch_size")
        if micro_bsz is not None and global_bsz is not None:
            micro_bsz = int(micro_bsz.decode())
            global_bsz = int(global_bsz.decode())
            new_dp = solve_dp(time_array, micro_bsz, global_bsz)
            # microbatch changed
            self.dp_version += 1
            # Publish plan and version together so training ranks cannot read
            # a new plan paired with the previous version (or vice versa).
            with self.client.pipeline(transaction=True) as pipeline:
                pipeline.set('batch_distribution', str(new_dp))
                pipeline.set('dp_version', self.dp_version)
                pipeline.execute()
        else:
            logging.warning(f"batch size not found: microbsz={micro_bsz}, globalbsz={global_bsz}")

    def adjust_pipeline_parallel(self,
                                 comm_cliques: Dict[int, CommunicatorClique],
                                 comm_tasks: List,
                                 comm_results: List[Dict[int, PerformanceMetric]]):
        max_comm_time_per_pp_stage = self.parse_communication_results(comm_results, comm_cliques)
        if not max_comm_time_per_pp_stage:
            logging.warning("Skipping PP repartition because no PP communication measurements were collected")
            return
        raw_total_num_layers = self.client.get("total_num_layers")
        if raw_total_num_layers is None:
            raise ValueError("Missing total_num_layers in Redis; cannot repartition the pipeline")
        if isinstance(raw_total_num_layers, bytes):
            raw_total_num_layers = raw_total_num_layers.decode()
        total_num_layers = int(raw_total_num_layers)
        layer_split = self.split_layers_by_stage_latency(
            total_num_layers, max_comm_time_per_pp_stage)
        apply_layer_split(
            self.client,
            layer_split,
            total_num_layers=total_num_layers,
            expected_stages=len(max_comm_time_per_pp_stage),
        )
        # pause the training process and invoke restart from memory
        self.client.set("terminate_ctl", 123)

    def find_slow_reason(self,
                         comp_results: Dict[int, PerformanceMetric],
                         comm_results: Dict[Tuple[int], List],
                         comm_cliques: Dict[int, CommunicatorClique]):
        comp_results = np.array([i.avg_lat for i in comp_results.values()])
        if np.max(comp_results) > 2 * np.median(comp_results):
            return "comp"
        comm_parsed = np.array(self.parse_communication_results(comm_results, comm_cliques))
        if len(comm_parsed) >= 2 and np.max(comm_parsed) > 1.1 * np.median(comm_parsed):
            return "comm"

    def mitigate_failslow(self,
                          dp_check_interval: int,
                          comp_results: Dict[int, PerformanceMetric],
                          comm_cliques: Dict[int, CommunicatorClique],
                          comm_tasks: Dict[Tuple[int], List],
                          comm_results: Dict[Tuple[int], List]):
        logging.info("Mitigating fail-slow problem...")
        slow_start = time.time()
        dp_adjusted, pp_adjusted = False, False
        slow_iter_time_init = float(self.client.get("cur_iter_time").decode())
        min_iter_time_init = float(self.client.get('min_iter_time').decode())
        reason = self.find_slow_reason(comp_results, comm_results, comm_cliques)
        logging.info(f"Reason of fail-slow is {reason}")
        if slow_iter_time_init <= min_iter_time_init * 1.1:
            self.adjust_batchsize_distribution(comp_results, comm_cliques)
            return

        while True:
            # normal iteration time
            min_iter_time = float(self.client.get('min_iter_time').decode())
            # iteration time after fail-slow
            slow_iter_time = float(self.client.get("cur_iter_time").decode())
            dp_cost = self.estimator.get_dp_adjustment_cost(dp_check_interval//2, min_iter_time, slow_iter_time)
            pp_cost = self.estimator.get_pp_adjustment_cost()
            time_since_slow = 1000 * (time.time() - slow_start)
            logging.info(f"[Mitigation Plan] DPcost={dp_cost}, PPcost={pp_cost}, time_since_slow={time_since_slow}, min_iter={min_iter_time}, slow_iter={slow_iter_time}")
            if (reason != 'comm') and (time_since_slow >= dp_cost) and (not dp_adjusted):
                logging.info("[Mitigation Plan] Adjust DP")
                self.adjust_batchsize_distribution(comp_results, comm_cliques)
                dp_adjusted = True
            if (reason == 'comm') and (time_since_slow >= pp_cost) and (not pp_adjusted):
                logging.info("[Mitigation Plan] Adjust PP")
                self.adjust_pipeline_parallel(comm_cliques, comm_tasks, comm_results)
                pp_adjusted = True
            if time_since_slow >= 2 * pp_cost:
                break
            time.sleep(1)
