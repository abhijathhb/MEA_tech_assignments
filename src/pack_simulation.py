# Imports
import numpy as np
import pandas as pd

def group_capacity(cells, group_indices):
    """Capacity of a parallel group ≈ sum of member cell capacities."""
    return cells.loc[group_indices, "Q"].sum()


def module_capacity(cells, module):
    """Module capacity for series groups = min(group capacities)."""
    caps = [group_capacity(cells, grp) for grp in module["series_groups"]]
    return min(caps)


def pack_capacity(cells, modules):
    """Pack capacity for modules in series = min(module capacities)."""
    m_caps = [module_capacity(cells, m) for m in modules]
    return min(m_caps)


def apply_balancing(cells, module, balance_strength=0.02, spread_threshold=0.02):
    """
    Very simplified balancing:
    - For each parallel group, check SoC spread.
    - If spread > threshold, 'bleed' the highest-SoC cells a bit
      (multiply Q by (1 - extra_deg)).
    """
    extra_loss_ah_total = 0.0

    for grp in module["series_groups"]:
        grp_cells = cells.loc[grp]

        SoC_max = grp_cells["SoC"].max()
        SoC_min = grp_cells["SoC"].min()
        spread = SoC_max - SoC_min

        if spread > spread_threshold:
            # Cells at highest SoC
            high_soc_idx = grp_cells[grp_cells["SoC"] == SoC_max].index
            extra_deg = balance_strength * spread  # stronger imbalance → more bleeding

            # Reduce capacity a small fraction to mimic extra cycling/heat
            old_Q = cells.loc[high_soc_idx, "Q"]
            cells.loc[high_soc_idx, "Q"] = old_Q * (1.0 - extra_deg)

            # Track approximate extra Ah loss
            extra_loss_ah_total += (old_Q * extra_deg).sum()

    return extra_loss_ah_total