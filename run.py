try:
    import google.colab # type: ignore
    from google.colab import output
    COLAB = True
    # %pip install sae-lens transformer-lens sae-dashboard
except:
    COLAB = False
    # from IPython import get_ipython # type: ignore
    # ipython = get_ipython(); assert ipython is not None
    # ipython.run_line_magic("load_ext", "autoreload")
    # ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd

# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# from huggingface_hub import notebook_login

def print_gpu_memory():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2  # Convert to MB
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**2  # Convert to MB
        
        print(f"GPU Memory Allocated: {memory_allocated:.2f} MB")
        print(f"GPU Memory Reserved: {memory_reserved:.2f} MB")
        print(f"Total GPU Memory: {total_memory:.2f} MB")
        # Calculate and print free memory
        free_memory = total_memory - memory_reserved
        print(f"Free GPU Memory: {free_memory:.2f} MB")

    else:
        print("CUDA is not available on this system.")

print_gpu_memory()

from huggingface_hub import HfApi, login
login(token="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")

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# TODO: Make this nicer.
df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
df # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model.


from sae_lens import SAE, HookedSAETransformer

model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)


# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
torch.cuda.empty_cache()
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "layer_20/width_16k/average_l0_71", # won't always be a hook point
    # sae_id = "layer_25/width_16k/average_l0_116", # won't always be a hook point
    device = device
)


from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)


token_dataset = tokenize_and_concatenate(
    dataset= dataset,# type: ignore
    tokenizer = model.tokenizer, # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)




sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
print(token_dataset.shape)
with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = token_dataset[:3]["tokens"]
    print(batch_tokens.shape)
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    feature_acts = sae.encode(cache[sae.cfg.hook_name])
    sae_out = sae.decode(feature_acts)

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l0", l0.mean().item())
    px.histogram(l0.flatten().cpu().numpy()).show()





from transformer_lens import utils
from functools import partial

example_prompt = "When John and Mary went to the shops, John gave the bag to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

logits, cache = model.run_with_cache(example_prompt, prepend_bos=True)
tokens = model.to_tokens(example_prompt)
sae_out = sae(cache[sae.cfg.hook_name])


def reconstr_hook(activations, hook, sae_out):
    return sae_out


def zero_abl_hook(mlp_out, hook):
    return torch.zeros_like(mlp_out)


hook_name = sae.cfg.hook_name

print("Orig", model(tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (
                hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_abl_hook)],
    ).item(),
)


with model.hooks(
    fwd_hooks=[
        (
            hook_name,
            partial(reconstr_hook, sae_out=sae_out),
        )
    ]
):
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


print_gpu_memory()



# instantiate an object to hold activations from a dataset
from sae_lens import ActivationsStore

# a convenient way to instantiate an activation store is to use the from_sae method
activation_store = ActivationsStore.from_sae(
    model=model,
    sae=sae,
    streaming=True,
    # fairly conservative parameters here so can use same for larger
    # models without running out of memory.
    store_batch_size_prompts=8,
    train_batch_size_tokens=4096,
    n_batches_in_buffer=32,
    device=device,
)
def list_flatten(nested_list):
    return [x for y in nested_list for x in y]

# A very handy function Neel wrote to get context around a feature activation
def make_token_df(tokens, len_prefix=5, len_suffix=3, model = model):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]
    
    context = []
    prompt = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            prompt.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        prompt=prompt,
        pos=pos,
        label=label,
    ))


# finding max activating examples is a bit harder. To do this we need to calculate feature activations for a large number of tokens
feature_list = torch.randint(0, sae.cfg.d_sae, (100,))
examples_found = 0
all_fired_tokens = []
all_feature_acts = []
all_reconstructions = []
all_token_dfs = []

total_batches = 100
batch_size_prompts = activation_store.store_batch_size_prompts
batch_size_tokens = activation_store.context_size * batch_size_prompts
pbar = tqdm(range(total_batches))
for i in pbar:
    tokens = activation_store.get_batch_tokens()
    tokens_df = make_token_df(tokens)
    tokens_df["batch"] = i
    
    flat_tokens = tokens.flatten()
    
    _, cache = model.run_with_cache(tokens, stop_at_layer = sae.cfg.hook_layer + 1, names_filter = [sae.cfg.hook_name])
    sae_in = cache[sae.cfg.hook_name]
    feature_acts = sae.encode(sae_in).squeeze()

    feature_acts = feature_acts.flatten(0,1)
    fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
    fired_tokens = model.to_str_tokens(flat_tokens[fired_mask])
    reconstruction = feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]

    token_df = tokens_df.iloc[fired_mask.cpu().nonzero().flatten().numpy()]
    all_token_dfs.append(token_df)
    all_feature_acts.append(feature_acts[fired_mask][:, feature_list])
    all_fired_tokens.append(fired_tokens)
    all_reconstructions.append(reconstruction)
    
    examples_found += len(fired_tokens)
    # print(f"Examples found: {examples_found}")
    # update description
    pbar.set_description(f"Examples found: {examples_found}")
    
# flatten the list of lists
all_token_dfs = pd.concat(all_token_dfs)
all_fired_tokens = list_flatten(all_fired_tokens)
all_reconstructions = torch.cat(all_reconstructions)
all_feature_acts = torch.cat(all_feature_acts)




from tqdm import tqdm
from functools import partial 

# def find_max_activation(model, sae, activation_store, feature_idx, num_batches=100):
#     '''
#     Find the maximum activation for a given feature index. This is useful for 
#     calibrating the right amount of the feature to add.
#     '''
#     max_activation = 0.0

#     pbar = tqdm(range(num_batches))
#     for _ in pbar:
#         tokens = activation_store.get_batch_tokens()
        
#         _, cache = model.run_with_cache(
#             tokens, 
#             stop_at_layer=sae.cfg.hook_layer + 1, 
#             names_filter=[sae.cfg.hook_name]
#         )
#         sae_in = cache[sae.cfg.hook_name]
#         feature_acts = sae.encode(sae_in).squeeze()

#         feature_acts = feature_acts.flatten(0, 1)
#         batch_max_activation = feature_acts[:, feature_idx].max().item()
#         max_activation = max(max_activation, batch_max_activation)
        
#         pbar.set_description(f"Max activation: {max_activation:.4f}")

#     return max_activation

def find_max_activations(model, sae, activation_store, feature_indices, num_batches=100):
    '''
    Find the maximum activation for multiple feature indices. This is useful for 
    calibrating the right amount of the features to add.
    '''
    max_activations = torch.zeros(len(feature_indices), device=model.cfg.device)

    pbar = tqdm(range(num_batches))
    for _ in pbar:
        tokens = activation_store.get_batch_tokens()
        
        _, cache = model.run_with_cache(
            tokens, 
            stop_at_layer=sae.cfg.hook_layer + 1, 
            names_filter=[sae.cfg.hook_name]
        )
        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        batch_max_activations = feature_acts[:, feature_indices].max(dim=0).values
        max_activations = torch.maximum(max_activations, batch_max_activations)
        
        # Update description with max activation for each feature
        desc = " | ".join([f"F{idx}: {act:.4f}" for idx, act in zip(feature_indices, max_activations)])
        pbar.set_description(f"Max activations: {desc}")

    return max_activations.tolist()


# def steering(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
#     # Note if the feature fires anyway, we'd be adding to that here.
#     return activations + max_act * steering_strength * steering_vector

def steering(activations, hook, steering_features, steering_strengths, max_acts):
    steering_vector = torch.zeros_like(activations[0])
    for feature, strength, max_act in zip(steering_features, steering_strengths, max_acts):
        steering_vector += max_act * strength * sae.W_dec[feature]
    return activations + steering_vector

# def generate_with_steering(model, sae, prompt, steering_feature, max_act, steering_strength=1.0, max_new_tokens=512):
#     input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    
#     steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)
    
#     steering_hook = partial(
#         steering,
#         steering_vector=steering_vector,
#         steering_strength=steering_strength,
#         max_act=max_act
#     )
    
#     # standard transformerlens syntax for a hook context for generation
#     with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
#         output = model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.9,
#             stop_at_eos = False if device == "mps" else True,
#             prepend_bos = sae.cfg.prepend_bos,
#         )
    
#     return model.tokenizer.decode(output[0])

# def generate_with_steering(model, sae, prompt, steering_features, max_acts, steering_strengths, max_new_tokens=95):
#     input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

#     steering_hook = partial(
#         steering,
#         steering_features=steering_features,
#         steering_strengths=steering_strengths,
#         max_acts=max_acts
#     )

#     with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
#         output = model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.9,
#             stop_at_eos = False if device == "mps" else True,
#             prepend_bos = sae.cfg.prepend_bos,
#         )

#     return model.tokenizer.decode(output[0])

def generate_with_steering(model, sae, prompt, steering_features, max_acts, steering_strengths, max_new_tokens=95):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

    steering_hook = partial(
        steering,
        steering_features=steering_features,
        steering_strengths=steering_strengths,
        max_acts=max_acts
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos = False if device == "mps" else True,
            prepend_bos = sae.cfg.prepend_bos,
        )

    return model.tokenizer.decode(output[0])

# Choose a feature to steer
bridge_steering_feature = 6492  # Choose a feature to steer towards
san_francisco_steering_feature = 3124

# bridge_steering_feature = 16057 # layer 25
# san_francisco_steering_feature = 4233 # layer 25

# Find the maximum activation for this feature
[bridge_max_act, san_francisco_max_act] = find_max_activations(model, sae, activation_store, [bridge_steering_feature, san_francisco_steering_feature])
print(f"Maximum activation for feature {bridge_steering_feature}: {bridge_max_act:.4f}")
print(f"Maximum activation for feature {san_francisco_steering_feature}: {san_francisco_max_act:.4f}")


# note we could also get the max activation from Neuronpedia (https://www.neuronpedia.org/api-doc#tag/lookup/GET/api/feature/{modelId}/{layer}/{index})
prompt1 = "Once upon a time"
prompt2 = "The most important structure in the world is"

max_new_tokens = 256 
for i in range(2):
    print("\n\n\n\nGenerating iterartion", i)
    print_gpu_memory()
    print("\n\n\n")
    # Generate text without steering for comparison
    normal_text1 = model.generate(
        prompt1,
        max_new_tokens=max_new_tokens, 
        stop_at_eos = False if device == "mps" else True,
        prepend_bos = sae.cfg.prepend_bos,
    )


    print("\n\n")
    print("\nNormal text (without steering):")
    print(normal_text1)

    normal_text2 = model.generate(
        prompt2,
        max_new_tokens=max_new_tokens, 
        stop_at_eos = False if device == "mps" else True,
        prepend_bos = sae.cfg.prepend_bos,
    )
    print('\n\n')
    print(normal_text2)

# feature_set = [san_francisco_steering_feature, bridge_steering_feature]
# feature_acts = [san_francisco_max_act, bridge_max_act]
# feature_strengths = [1.0, 1.0]
feature_set = [bridge_steering_feature]
feature_acts = [bridge_max_act]
feature_strengths = [2.0]

for i in range(5):
    # Generate text with steering
    # steered_text = generate_with_steering(model, sae, prompt, [san_francisco_steering_feature, bridge_steering_feature], [san_francisco_max_act, bridge_max_act], steering_strengths=[0.3, 1], max_new_tokens=max_new_tokens)
    steered_text1 = generate_with_steering(model, sae, prompt1, feature_set, feature_acts, steering_strengths=feature_strengths, max_new_tokens=max_new_tokens)

    print("Steered text:")
    print(steered_text1)

    steered_text2 = generate_with_steering(model, sae, prompt2, feature_set, feature_acts, steering_strengths=feature_strengths, max_new_tokens=max_new_tokens)

    print("Steered text:")
    print(steered_text2)
