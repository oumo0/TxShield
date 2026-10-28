from typing import Any

from txshield.common.trace_utils import child_call_key

def _function_name(node: dict) -> Any:
    return node.get("function", node.get("func"))


def extract_deepest_chain(root: dict, target_function: str) -> tuple[list[dict], int | None]:
    deepest: list[dict] = []
    deepest_target_depth: int | None = None

    def dfs(node: dict, depth: int, path: list[dict]) -> None:
        nonlocal deepest, deepest_target_depth
        if not isinstance(node, dict):
            return
        current_path = path + [node]
        if _function_name(node) == target_function:
            if len(current_path) > len(deepest):
                deepest = current_path
                deepest_target_depth = int(node.get("depth", depth))
        children = node.get(child_call_key(node), [])
        for child in children if isinstance(children, list) else []:
            dfs(child, depth + 1, current_path)

    dfs(root, int(root.get("depth", 0)), [])
    return deepest, deepest_target_depth

def prune_same_level_non_target(
    node: dict,
    target_depth: int,
    target_function: str,
    current_depth: int,
) -> None:
    children_key = child_call_key(node)
    children = node.get(children_key)
    if not isinstance(children, list):
        return
    kept = []
    for child in children:
        if not isinstance(child, dict):
            continue
        child_depth = int(child.get("depth", current_depth + 1))
        if _function_name(child) == target_function or child_depth != target_depth:
            kept.append(child)
            prune_same_level_non_target(
                child,
                target_depth,
                target_function,
                child_depth,
            )
    node[children_key] = kept

def prune_call_tree(root: dict, target_function: str) -> dict:
    chain, target_depth = extract_deepest_chain(root, target_function)
    if chain and target_depth is not None:
        selected_root = chain[0]
        root_depth = int(selected_root.get("depth", 0))
        prune_same_level_non_target(
            selected_root,
            target_depth,
            target_function,
            root_depth,
        )
        return selected_root
    return root

def select_roots(data: dict) -> tuple[str | None, list]:
    if isinstance(data.get("logs"), dict) and data["logs"].get("calls"):
        return "logs.calls", data["logs"]["calls"]
    if data.get("calls"):
        return "calls", data["calls"]
    if data.get("function_call"):
        return "function_call", data["function_call"]
    return None, []

def prune_data(data: Any, target_function: str = "executeOperation",
               config: dict | None = None) -> Any:
    from txshield.preprocessing.pruning.rules import apply_general_rules
    if not isinstance(data, dict):
        return data
    key, roots = select_roots(data)
    if not key:
        return data
    if not isinstance(roots, list) or any(not isinstance(root, dict) for root in roots):
        raise ValueError("Trace roots must be a list of call objects")

    # Algorithm 1 accepts one tree. Apply it independently to each tree in
    # the transaction's root list, preserving order and all surviving roots.
    pruned_roots = []
    for root in roots:
        root = apply_general_rules(root, config=config)
        if root is not None:
            pruned_roots.append(prune_call_tree(root, target_function))
    if key == "logs.calls":
        data["logs"]["calls"] = pruned_roots
    else:
        data[key] = pruned_roots
    return data
