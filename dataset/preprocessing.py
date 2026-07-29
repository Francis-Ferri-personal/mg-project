def find_jumps(target_list: list) -> list[tuple[str, int]]:
    """Detect target jumps (±15 values). Iterates from i=0; if target[i] is ±15 and delta from i-1 ≥ 15, it's a jump."""
    jumps_ids = []
    for i in range(len(target_list)):
        if i == 0:
            if abs(target_list[0]) == 15:
                direction = "positive" if target_list[0] > 0 else "negative"
                jumps_ids.append((direction, 0))
        else:
            if abs(target_list[i]) == 15 and abs(target_list[i] - target_list[i - 1]) >= 15:
                diff = target_list[i] - target_list[i - 1]
                direction = "positive" if diff > 0 else "negative"
                jumps_ids.append((direction, i))

    return jumps_ids


def analyse_jumps(jump_ids: list[tuple[str, int]]) -> dict:
    if not jump_ids:
        return {"cycles": []}

    same_dir_indices = [
        idx for direction, idx in jump_ids if direction == "positive"
    ]
    cycles = [
        (same_dir_indices[i], same_dir_indices[i + 1])
        for i in range(len(same_dir_indices) - 1)
    ]

    return cycles


def get_cycles(target_list: list) -> list[tuple[int, int]]:
    """
    Convenience: find jumps and extract cycles in one call.
    Returns list of (start_idx, end_idx).
    """
    jump_ids = find_jumps(target_list)
    cycles = analyse_jumps(jump_ids)
    
    return cycles
