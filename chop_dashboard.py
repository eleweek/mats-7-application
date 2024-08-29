import json

from copy import deepcopy
from pathlib import Path

from dataclasses import is_dataclass, fields
from typing import get_origin, get_args


from transformer_lens import HookedTransformer
from sae_lens import SAE


from sae_dashboard.sae_vis_data import SaeVisConfig


from sae_dashboard.html_fns import HTML
from sae_dashboard.sae_vis_data import SaeVisData
from sae_dashboard.utils_fns import get_decode_html_safe_fn
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.html_fns import HTML

def save_feature_centric_vis(
    cfg,
    model,
    feature_data_dict,
    filename,
) -> None:
    iterator = list(feature_data_dict.items())

    HTML_OBJ = HTML()  # Initialize HTML object for combined file

    # For each FeatureData object, we get the html_obj for its feature-centric vis
    for feature, feature_data in iterator:
        decode_fn = get_decode_html_safe_fn(model.tokenizer)
        html_obj = feature_data._get_html_data_feature_centric(
            cfg.feature_centric_layout, decode_fn
        )

        feature_HTML_OBJ = HTML()  # Initialize a new HTML object for each feature
        feature_HTML_OBJ.js_data[str(feature)] = deepcopy(html_obj.js_data)
        feature_HTML_OBJ.html_data = deepcopy(html_obj.html_data)

        # Add the aggdata
        feature_HTML_OBJ.js_data = {
            "AGGDATA": {},
            "DASHBOARD_DATA": feature_HTML_OBJ.js_data,
        }

        # Generate filename for this feature
        feature_filename = Path(filename).with_stem(
            f"{Path(filename).stem}_feature_{feature}.html"
        )

        # Save the HTML for this feature
        feature_HTML_OBJ.get_html(
            layout_columns=cfg.feature_centric_layout.columns,
            layout_height=cfg.feature_centric_layout.height,
            filename=feature_filename,
            first_key=str(feature),
        )

from dataclasses import is_dataclass, fields

def json_to_feature_data(json_data, target_class):
    if is_dataclass(target_class):
        if isinstance(json_data, dict):
            kwargs = {}
            for field in fields(target_class):
                if field.name in json_data:
                    field_value = json_data[field.name]
                    field_type = field.type
                    kwargs[field.name] = convert_field(field_value, field_type)
            return target_class(**kwargs)
        else:
            raise ValueError(f"Expected dict for {target_class}, got {type(json_data)}")
    else:
        return json_data

def convert_field(field_value, field_type):
    origin = get_origin(field_type)
    if origin is None:
        if is_dataclass(field_type):
            return json_to_feature_data(field_value, field_type)
        return field_value
    elif origin is list:
        element_type = get_args(field_type)[0]
        return [convert_field(item, element_type) for item in field_value]
    elif origin is dict:
        key_type, value_type = get_args(field_type)
        return {convert_field(k, key_type): convert_field(v, value_type) for k, v in field_value.items()}
    else:
        return field_value



with open("feature_data_dict.json") as f:
    feature_data_dict = json.load(f)

parsed = {}
for feature, feature_data_raw in feature_data_dict.items():
    parsed[feature] = json_to_feature_data(feature_data_raw, FeatureData)

device = "cuda"
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res", # see other options in sae_lens/pretrained_saes.yaml
    sae_id="layer_20/width_65k/average_l0_61",
    # sae_id = "layer_20/width_16k/average_l0_71", # won't always be a hook point
    # sae_id = "layer_25/width_16k/average_l0_116", # won't always be a hook point
    device = device
)

config = SaeVisConfig(
    hook_point=sae.cfg.hook_name,
    features=range(65536),
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

save_feature_centric_vis(config, model, parsed, "individual_dashboards/chop")
