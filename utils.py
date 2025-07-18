def copy_weights(source_state_dict, target_state_dict):
    for src_key in source_state_dict.keys():
        if src_key.startswith("base_model"):
            tgt_key = src_key.replace("base_model.", "")
        else:
            tgt_key = src_key
        if (
            src_key.startswith("vision_tower")
            or src_key.startswith("base_model.vision_tower")
        ) and tgt_key in target_state_dict:
            target_state_dict[tgt_key] = source_state_dict[src_key]
        else:
            print(f"Skipping source {src_key}")

    return target_state_dict
