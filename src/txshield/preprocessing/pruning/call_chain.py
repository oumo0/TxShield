from typing import Any

def extract_deepest_chain(root: dict, target_function: str) -> tuple[list[dict], int | None]:
    deepest: list[dict] = []
    first_depth: int | None = None

    def dfs(node: dict, depth: int, path: list[dict]) -> None:
        nonlocal deepest, first_depth
        if not isinstance(node, dict):
            return
        current_path = path + [node]
        if node.get("function") == target_function:
            if first_depth is None:
                first_depth = node.get("depth", depth)
            if len(current_path) > len(deepest):
                deepest = current_path
        for child in node.get("internal_calls", []) if isinstance(node.get("internal_calls", []), list) else []:
            dfs(child, depth + 1, current_path)

    dfs(root, int(root.get("depth", 0)), [])
    return deepest, first_depth

def prune_same_level_non_target(node: dict, target_depth: int, target_function: str) -> None:
    children = node.get("internal_calls")
    if not isinstance(children, list):
        return
    kept = []
    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("function") == target_function or child.get("depth") != target_depth:
            kept.append(child)
            prune_same_level_non_target(child, target_depth, target_function)
    node["internal_calls"] = kept

def prune_call_tree(root: dict, target_function: str) -> dict:
    chain, first_depth = extract_deepest_chain(root, target_function)
    if chain and first_depth is not None:
        selected_root = chain[0]
        prune_same_level_non_target(selected_root, first_depth, target_function)
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
    root = roots[0] if isinstance(roots, list) and roots else roots
    root = apply_general_rules(root, config=config)
    if root is None:
        root = {"_pruned": "all nodes removed"}
    root = prune_call_tree(root, target_function)
    if key == "logs.calls":
        data["logs"]["calls"] = [root]
    else:
        data[key] = [root]
    return data
