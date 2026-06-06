"""Query the Flux allocation and map it to PotMill's per-stage executor worker counts."""

from dataclasses import dataclass


@dataclass
class Resources:
    nnodes: int
    ncores: int
    ngpus: int
    gpus_per_node: int
    n_label_workers: int
    n_fit_workers: int
    n_entropy_workers: int
    threads_per_worker: int
    n_featurize_workers: int


def query_flux():
    """Return (resource_status, ncores, ngpus, nnodes) for the current Flux allocation."""
    import flux
    import flux.resource

    handle = flux.Flux()
    rs = flux.resource.status.ResourceStatusRPC(handle).get()
    rl = flux.resource.list.resource_list(handle).get()
    nnodes = len(list(rs.nodelist))
    print("NODELIST:", rs.nodelist, " #CORES:", rl.all.ncores, " #GPUS:", rl.all.ngpus, flush=True)
    return rs, rl.all.ncores, rl.all.ngpus, nnodes


def worker_layout(config, nnodes, ncores, ngpus):
    """Map the allocation to per-stage worker counts: the labeling/fitting GPU split (config-driven,
    fit_gpus_per_node GPUs/node for fitting and the rest for labeling) plus entropy and featurize
    workers. n_entropy_workers and featurize_workers_per_node are per-node knobs scaled by nnodes."""
    gpus_per_node = ngpus // nnodes if nnodes else 0
    fit_gpus_per_node = config["ourFit"]["fit_gpus_per_node"]
    assert 0 < fit_gpus_per_node < gpus_per_node, (
        f"fit_gpus_per_node ({fit_gpus_per_node}) must be >0 and leave GPUs for labeling "
        f"(gpus_per_node={gpus_per_node})"
    )
    n_entropy_workers = config["ourStructureGen"].get("n_entropy_workers", 1) * nnodes
    return Resources(
        nnodes=nnodes,
        ncores=ncores,
        ngpus=ngpus,
        gpus_per_node=gpus_per_node,
        n_label_workers=(gpus_per_node - fit_gpus_per_node) * nnodes,
        n_fit_workers=fit_gpus_per_node * nnodes,
        n_entropy_workers=n_entropy_workers,
        threads_per_worker=max(1, 32 // n_entropy_workers),
        n_featurize_workers=config["ourFeaturization"]["featurize_workers_per_node"] * nnodes,
    )
