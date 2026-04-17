"""
Project-Specific Export Adapters.

Each adapter provides model loading and export customization for a specific project.
"""

from .pinn_lite_adapter import (
    load_model as load_pinn_lite_model,
    get_default_config as get_pinn_lite_config,
    validate_against_original as validate_pinn_lite,
)

from .cloth4d_adapter import (
    load_model as load_cloth4d_model,
    get_default_config as get_cloth4d_config,
    wrap_for_export as wrap_cloth4d,
    generate_query_grid,
    verify_sine_export,
)

from .cloth_gnn_adapter import (
    load_model as load_cloth_gnn_model,
    get_default_config as get_cloth_gnn_config,
    wrap_for_export as wrap_cloth_gnn,
    create_template_mesh,
    export_with_template_mesh,
    ClothGNNExportable,
    DenseMessagePassing,
)

from .coastflow_adapter import (
    load_model as load_coastflow_model,
    get_default_config as get_coastflow_config,
    create_lookup_model as create_coastflow_lookup,
    generate_scenario_cache,
    create_coastline_mask,
    export_with_coastline,
    CoastFlowExportable,
    ScenarioLookupExportable,
)

__all__ = [
    # PINN Lite (Project 15)
    "load_pinn_lite_model",
    "get_pinn_lite_config",
    "validate_pinn_lite",
    # Cloth4D (Project 13)
    "load_cloth4d_model",
    "get_cloth4d_config",
    "wrap_cloth4d",
    "generate_query_grid",
    "verify_sine_export",
    # ClothGNN (Project 05)
    "load_cloth_gnn_model",
    "get_cloth_gnn_config",
    "wrap_cloth_gnn",
    "create_template_mesh",
    "export_with_template_mesh",
    "ClothGNNExportable",
    "DenseMessagePassing",
    # CoastFlow (Project 06)
    "load_coastflow_model",
    "get_coastflow_config",
    "create_coastflow_lookup",
    "generate_scenario_cache",
    "create_coastline_mask",
    "export_with_coastline",
    "CoastFlowExportable",
    "ScenarioLookupExportable",
]
