import redis
import logging
from typing import List, Optional


def apply_layer_split(client: redis.StrictRedis,
                      layer_split: List[int],
                      total_num_layers: Optional[int] = None,
                      expected_stages: Optional[int] = None):
    """Publish a complete PP repartition atomically.

    Transformer constructors read the offset and layer count independently, so
    publishing keys one at a time can expose a partially updated assignment to
    a restarting rank. Validate the plan before writing, then commit all stage
    assignments in one Redis transaction.
    """
    if not layer_split:
        raise ValueError("Pipeline layer split cannot be empty")
    if any(isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0
           for layers in layer_split):
        raise ValueError(f"Every pipeline stage must receive a positive integer layer count: {layer_split}")
    if expected_stages is not None and len(layer_split) != expected_stages:
        raise ValueError(
            f"Layer split has {len(layer_split)} stages, expected {expected_stages}: {layer_split}"
        )
    if total_num_layers is not None and sum(layer_split) != total_num_layers:
        raise ValueError(
            f"Layer split sums to {sum(layer_split)}, expected {total_num_layers}: {layer_split}"
        )

    offset = 0
    with client.pipeline(transaction=True) as pipeline:
        for stage, num_layers in enumerate(layer_split):
            pipeline.set(f"pp_offset_{stage}", offset)
            pipeline.set(f"pp_num_layers_{stage}", num_layers)
            offset += num_layers
        pipeline.execute()
    logging.critical(f"Apply layer split {layer_split} done")


if __name__ == '__main__':
    client = redis.StrictRedis("localhost", 6379)
    layer_split = [8, 5, 10, 9]
    apply_layer_split(client, layer_split)
