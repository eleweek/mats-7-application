import json
import torch
import dataclasses
from sae_lens import SAE
from sae_lens.evals import EvalConfig
from transformer_lens import HookedTransformer

from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner

from huggingface_hub import HfApi, login
login(token="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")



# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res", # see other options in sae_lens/pretrained_saes.yaml
    sae_id="layer_20/width_65k/average_l0_61",
    # sae_id = "layer_20/width_16k/average_l0_71", # won't always be a hook point
    # sae_id = "layer_25/width_16k/average_l0_116", # won't always be a hook point
    device = device
)
# fold w_dec norm so feature activations are accurate
sae.fold_W_dec_norm()

from sae_lens import ActivationsStore

activations_store = ActivationsStore.from_sae(
    model=model,
    sae=sae,
    streaming=True,
    store_batch_size_prompts=16,
    n_batches_in_buffer=8,
    device=device,
)

# Some SAEs will require we estimate the activation norm and fold it into the weights. This is easy with SAE Lens.
if sae.cfg.normalize_activations == "expected_average_only_in":
    norm_scaling_factor = activations_store.estimate_norm_scaling_factor(
        n_batches_for_norm_estimate=30
    )
    sae.fold_activation_norm_scaling_factor(norm_scaling_factor)


from tqdm import tqdm


def get_tokens(
    activations_store: ActivationsStore,
    n_prompts: int,
):
    all_tokens_list = []
    pbar = tqdm(range(n_prompts))
    for _ in pbar:
        batch_tokens = activations_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens


token_dataset = get_tokens(activations_store, 100)

from sae_lens import run_evals

def do_evals():
    eval_metrics = run_evals(
        sae=sae,
        activation_store=activations_store,
        model=model,
        eval_config=EvalConfig(
            n_eval_reconstruction_batches=3, 
            batch_size_prompts=3, 
            compute_sparsity_metrics=True, 
            compute_variance_metrics=True, 
            compute_l2_norms=True, 
            compute_kl=True, compute_ce_loss=True)
    )

    print(eval_metrics)

    print("CE loss score with ablation", eval_metrics["metrics/ce_loss_with_ablation"])
    # CE Loss score should be high for residual stream SAEs, originally metrics/CE_loss_score"
    print("CE loss score", eval_metrics["metrics/ce_loss_score"])


    # ce loss without SAE should be fairly low < 3.5 suggesting the Model is being run correctly
    print("CE loss without SAE", eval_metrics["metrics/ce_loss_without_sae"])

    # ce loss with SAE shouldn't be massively higher
    print("CE loss with SAE", eval_metrics["metrics/ce_loss_with_sae"])


do_evals()

import gc

gc.collect()
torch.cuda.empty_cache()



from pathlib import Path


test_feature_idx_gpt = list(range(65536))
# test_feature_idx_gpt = list(range(6480, 6550))
# test_feature_idx_gpt = list(range(10))


feature_vis_config_gpt = SaeVisConfig(
    hook_point=sae.cfg.hook_name,
    features=test_feature_idx_gpt,
    # minibatch_size_features=64,
    # minibatch_size_tokens=256,  # this is number of prompts at a time.
    minibatch_size_features=32,
    minibatch_size_tokens=200,  # this is number of prompts at a time.
    verbose=True,
    device="cuda",
    cache_dir=Path(
        "demo_activations_cache"
    )  # this will enable us to skip running the model for subsequent features.
)

data = SaeVisRunner(feature_vis_config_gpt).run(
    encoder=sae,  # type: ignore
    model=model,
    tokens=token_dataset,
)



from sae_dashboard.data_writing_fns import save_feature_centric_vis

filename = f"demo_feature_dashboards.html"
save_feature_centric_vis(sae_vis_data=data, filename=filename)


class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

with open("feature_data_dict.json", "w") as f:
    json.dump(data.feature_data_dict, f, cls=EnhancedJSONEncoder)
