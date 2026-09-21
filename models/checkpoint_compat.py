"""Compatibility helpers for validated pre-publication checkpoints."""


def convert_legacy_rarn_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(
            "local_count_tree.evidence_projector.", "oca.response_projector.")
        new_key = new_key.replace(
            "local_count_tree.ordinal_head.", "oca.ordinal_head.")
        new_key = new_key.replace(
            "local_count_tree.shape_head.", "oca.allocation_head.")
        converted[new_key] = value
    return converted
