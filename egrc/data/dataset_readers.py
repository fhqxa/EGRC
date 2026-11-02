import os
import os.path as osp


IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp")


def _dedupe(paths):
    unique = []
    seen = set()
    for path in paths:
        normalized = osp.normpath(path)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _expand_base_roots(base_root):
    candidates = [base_root]
    normalized = osp.normpath(base_root)
    parent = osp.dirname(normalized)
    leaf = osp.basename(normalized)

    if leaf == "datasets":
        candidates.append(osp.join(parent, "dataset"))
    elif leaf == "dataset":
        candidates.append(osp.join(parent, "datasets"))

    return _dedupe(candidates)


def build_root_candidates(base_root, aliases, include_nested=True):
    base_roots = _expand_base_roots(base_root)
    candidates = []
    for root in base_roots:
        candidates.extend(osp.join(root, alias) for alias in aliases)
    candidates.extend(base_roots)
    if include_nested:
        nested_candidates = []
        for candidate in candidates:
            nested_candidates.extend(osp.join(candidate, alias) for alias in aliases)
        candidates.extend(nested_candidates)
    return _dedupe(candidates)


def resolve_dataset_root(base_root, aliases, validator=None):
    candidates = build_root_candidates(base_root, aliases)

    if validator is not None:
        for candidate in candidates:
            if osp.isdir(candidate) and validator(candidate):
                return candidate

    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]


def list_image_files(class_dir):
    return sorted(
        file_name
        for file_name in os.listdir(class_dir)
        if file_name.lower().endswith(IMAGE_EXTENSIONS)
    )
