"""
Enhanced Modularity Visualization
=================================
Advanced modularity-based brain connectivity visualization with PC classification.
Version 4 features: Camera controls, multiple views, enhanced node borders.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import colorsys
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Import from package modules
from .camera import CameraController
from .mesh import load_mesh_file
from .loaders import NetNeurotoolsModularityLoader
from .utils import (
    NumpyEncoder,
    classify_node_role,
    calculate_node_size,
    calculate_edge_width,
    filter_edges_by_module,
    threshold_matrix_top_n
)


def create_enhanced_modularity_visualization(
    vertices: np.ndarray,
    faces: np.ndarray,
    roi_coords_df: pd.DataFrame,
    connectivity_matrix: np.ndarray,
    module_data: Dict,
    metrics_df: pd.DataFrame = None,
    plot_title: str = "Enhanced Modularity Visualization",
    save_path: str = "enhanced_modularity_viz.html",
    visualization_type: str = "all",
    node_sizing_mode: str = "both",
    base_node_size: int = 12,
    max_node_multiplier: float = 2.0,
    n_top_edges: Optional[int] = None,
    edge_width_range: Tuple[float, float] = (1.0, 6.0),
    mesh_color: str = 'lightgray',
    mesh_opacity: float = 0.15,
    show_labels: bool = True,
    label_font_size: int = 10,
    show_significance: bool = True,
    border_width: int = 6,
    actual_state_label: Optional[int] = None,
    camera_view: str = 'oblique',
    custom_camera: Optional[Dict] = None,
    enable_camera_controls: bool = True,
    save_all_views: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Create enhanced modularity visualization with camera controls.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices array
    faces : np.ndarray
        Mesh faces array
    roi_coords_df : pd.DataFrame
        ROI coordinates dataframe
    connectivity_matrix : np.ndarray
        Connectivity matrix
    module_data : dict
        Module assignment and metrics data
    metrics_df : pd.DataFrame, optional
        Node metrics dataframe
    plot_title : str
        Title for the plot
    save_path : str
        Path to save HTML file
    visualization_type : str
        Type: 'all', 'intra', 'inter', 'nodes_only', 'significant_only'
    node_sizing_mode : str
        Sizing mode: 'pc', 'zscore', 'both'
    base_node_size : int
        Base node size
    max_node_multiplier : float
        Maximum size multiplier
    n_top_edges : int, optional
        Number of top edges to keep
    edge_width_range : tuple
        (min_width, max_width) for edges
    mesh_color : str
        Brain mesh color
    mesh_opacity : float
        Brain mesh opacity
    show_labels : bool
        Whether to show ROI labels
    label_font_size : int
        Font size for labels
    show_significance : bool
        Whether to show significance indicators
    border_width : int
        Node border width in pixels
    actual_state_label : int, optional
        State label for display
    camera_view : str
        Camera view preset name
    custom_camera : dict, optional
        Custom camera position
    enable_camera_controls : bool
        Whether to enable camera controls
    save_all_views : bool
        Whether to save all standard views

    Returns
    -------
    tuple
        (plotly.Figure, stats_dict)
    """

    print(f"Creating enhanced {visualization_type} visualization (node size: {node_sizing_mode})...")

    # Extract module assignments
    module_assignments = module_data['consensus']
    if len(module_assignments.shape) > 1:
        module_assignments = module_assignments.flatten()

    # Get PC and Z-score data
    if metrics_df is not None and 'participation_coef' in metrics_df.columns:
        pc_values = metrics_df['participation_coef'].values
        z_scores = metrics_df['within_module_zscore'].values
        roi_names = metrics_df['roi_name'].values
    else:
        pc_values = module_data.get('participation_coef', np.zeros(len(module_assignments)))
        z_scores = module_data.get('within_module_zscore', np.zeros(len(module_assignments)))
        roi_names = [f"ROI_{i}" for i in range(len(module_assignments))]

    # Classify nodes
    node_roles = []
    node_role_colors = []
    for pc, z in zip(pc_values, z_scores):
        role, color = classify_node_role(z, pc)
        node_roles.append(role)
        node_role_colors.append(color)

    # Calculate dynamic node sizes
    node_sizes = []
    for pc, z in zip(pc_values, z_scores):
        size = calculate_node_size(pc, z, node_sizing_mode, base_node_size, max_node_multiplier)
        node_sizes.append(size)

    # Apply top N edges thresholding if requested
    if n_top_edges is not None:
        connectivity_matrix = threshold_matrix_top_n(connectivity_matrix, n_top_edges)

    # Get module significance
    if 'module_significance' in module_data and show_significance:
        module_significance = module_data['module_significance']
    else:
        module_significance = np.ones(len(np.unique(module_assignments)), dtype=bool)

    # Identify active nodes
    node_has_connection = (np.any(connectivity_matrix != 0, axis=0) |
                           np.any(connectivity_matrix != 0, axis=1))
    active_nodes = np.where(node_has_connection)[0]

    # Generate module colors
    unique_modules = np.unique(module_assignments[active_nodes])
    n_modules = len(unique_modules)

    if n_modules <= 10:
        colors = px.colors.qualitative.Plotly[:n_modules]
    else:
        colors = []
        for i in range(n_modules):
            hue = i / n_modules
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')

    module_to_color = {module: colors[i] for i, module in enumerate(unique_modules)}

    # Create figure
    fig = go.Figure()

    # Add brain mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=mesh_opacity,
        color=mesh_color,
        name='Brain Surface',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Get all edge weights for normalization
    all_weights = connectivity_matrix[connectivity_matrix != 0]

    # Process each module separately
    for module_id in unique_modules:
        module_nodes = [i for i in active_nodes if module_assignments[i] == module_id]

        if not module_nodes:
            continue

        # Get module significance
        module_idx = np.where(np.unique(module_assignments) == module_id)[0][0]
        is_significant = module_significance[module_idx] if show_significance and module_idx < len(module_significance) else True

        module_name = f'Module {int(module_id)}'
        if is_significant and show_significance:
            module_name += '*'

        # Filter edges based on visualization type
        if visualization_type == "nodes_only":
            module_matrix = np.zeros_like(connectivity_matrix)
        elif visualization_type == "intra":
            module_matrix = filter_edges_by_module(connectivity_matrix, module_assignments, module_id, 'intra')
        elif visualization_type == "inter":
            module_matrix = filter_edges_by_module(connectivity_matrix, module_assignments, module_id, 'inter')
        elif visualization_type == "significant_only":
            if is_significant:
                module_matrix = connectivity_matrix.copy()
                for i in range(len(module_assignments)):
                    if module_assignments[i] != module_id:
                        for j in range(len(module_assignments)):
                            if module_assignments[j] != module_id:
                                module_matrix[i, j] = 0
            else:
                module_matrix = np.zeros_like(connectivity_matrix)
        else:  # "all"
            module_matrix = connectivity_matrix.copy()

        # Add edges with variable width
        for i in module_nodes:
            for j in range(i + 1, connectivity_matrix.shape[0]):
                if module_matrix[i, j] != 0 and j in active_nodes:
                    edge_width = calculate_edge_width(
                        module_matrix[i, j],
                        all_weights,
                        edge_width_range[0],
                        edge_width_range[1]
                    )

                    edge_trace = go.Scatter3d(
                        x=[roi_coords_df.loc[i, 'cog_x'], roi_coords_df.loc[j, 'cog_x']],
                        y=[roi_coords_df.loc[i, 'cog_y'], roi_coords_df.loc[j, 'cog_y']],
                        z=[roi_coords_df.loc[i, 'cog_z'], roi_coords_df.loc[j, 'cog_z']],
                        mode='lines',
                        line=dict(
                            color=module_to_color[module_id],
                            width=edge_width
                        ),
                        opacity=0.7,
                        hoverinfo='text',
                        hovertext=f"{roi_names[i]} <-> {roi_names[j]}<br>Strength: {module_matrix[i, j]:.4f}",
                        showlegend=False,
                        legendgroup=f'module_{module_id}'
                    )
                    fig.add_trace(edge_trace)

        # Get node properties for this module
        node_x = [roi_coords_df.loc[i, 'cog_x'] for i in module_nodes]
        node_y = [roi_coords_df.loc[i, 'cog_y'] for i in module_nodes]
        node_z = [roi_coords_df.loc[i, 'cog_z'] for i in module_nodes]

        module_node_sizes = [node_sizes[i] for i in module_nodes]
        module_node_roles = [node_roles[i] for i in module_nodes]
        module_node_colors = [node_role_colors[i] for i in module_nodes]
        module_roi_names = [roi_names[i] for i in module_nodes]

        # Create hover text
        hover_texts = []
        for i, node_idx in enumerate(module_nodes):
            hover_text = (
                f"<b>{module_roi_names[i]}</b><br>"
                f"Module: {int(module_id)}{' (SIGNIFICANT)' if is_significant and show_significance else ''}<br>"
                f"Role: {module_node_roles[i]}<br>"
                f"PC: {pc_values[node_idx]:.3f}<br>"
                f"Z-score: {z_scores[node_idx]:.3f}<br>"
                f"Node size: {module_node_sizes[i]:.1f}"
            )
            hover_texts.append(hover_text)

        # LAYER 1 - Border layer (role color)
        border_sizes = [size + border_width for size in module_node_sizes]
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=border_sizes,
                color=module_node_colors,
                opacity=0.95 if is_significant else 0.7,
            ),
            hoverinfo='skip',
            showlegend=False,
            legendgroup=f'module_{module_id}_border'
        ))

        # LAYER 2 - Inner node (module color)
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=module_node_sizes,
                color=module_to_color[module_id],
                opacity=0.9 if is_significant else 0.6,
                line=dict(
                    color=module_node_colors,
                    width=2
                )
            ),
            text=module_roi_names if show_labels else None,
            textposition='top center',
            textfont=dict(
                size=label_font_size,
                color='black',
                family='Arial'
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=True,
            name=module_name,
            legendgroup=f'module_{module_id}'
        ))

    # Set camera position
    if custom_camera is not None:
        camera = custom_camera
    else:
        camera = CameraController.get_camera_position(camera_view)

    # Update title
    if actual_state_label is not None:
        state_text = f"State {actual_state_label}"
    else:
        state_text = ""

    if 'Q_total' in module_data and module_data['Q_total'] != 0:
        q_score = module_data['Q_total']
        z_score = module_data.get('Q_z_score', 0)
        plot_title += f"<br>{state_text} - Q={q_score:.3f}, Z={z_score:.2f}"
    elif state_text:
        plot_title += f"<br>{state_text}"

    # Add camera view to title
    if 'name' in camera:
        plot_title += f"<br><i>View: {camera['name']}</i>"

    # Create role legend text
    role_legend_text = (
        "<b>Node Roles (border color):</b><br>"
        "* Ultra-peripheral (light gray)<br>"
        "* Peripheral (gray)<br>"
        "* Kinless (pink)<br>"
        "* Satellite Connector (blue)<br>"
        "* Provincial Hub (gold)<br>"
        "* Connector Hub (red-orange)"
    )

    # Camera control instructions
    camera_control_text = ""
    if enable_camera_controls:
        camera_control_text = (
            "<b>Camera Controls:</b><br>"
            "* Drag to rotate<br>"
            "* Scroll to zoom<br>"
            "* Right-click drag to pan<br>"
            "* Double-click to reset"
        )

    # Update layout with camera
    scene_dict = dict(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showgrid=False, zeroline=False, visible=False),
        bgcolor='white',
        camera=dict(
            eye=camera['eye'],
            center=camera['center'],
            up=camera['up']
        ),
        dragmode='orbit',
        aspectmode='data'
    )

    # Add camera preset buttons if enabled
    updatemenus = []
    if enable_camera_controls:
        camera_buttons = []
        for view_key, view_data in CameraController.PRESET_VIEWS.items():
            if view_key != 'custom':
                camera_buttons.append(
                    dict(
                        args=[{'scene.camera': {
                            'eye': view_data['eye'],
                            'center': view_data['center'],
                            'up': view_data['up']
                        }}],
                        label=view_data['name'],
                        method='relayout'
                    )
                )

        updatemenus = [
            dict(
                type='dropdown',
                showactive=True,
                buttons=camera_buttons,
                x=0.01,
                xanchor='left',
                y=0.99,
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            )
        ]

    fig.update_layout(
        scene=scene_dict,
        width=1200,
        height=900,
        title={
            'text': plot_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.85,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        updatemenus=updatemenus,
        annotations=[
            dict(
                text=role_legend_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.50,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            ),
            dict(
                text=camera_control_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.30,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)' if camera_control_text else 'rgba(255,255,255,0)',
                bordercolor='black' if camera_control_text else 'rgba(0,0,0,0)',
                borderwidth=1 if camera_control_text else 0,
                borderpad=4
            ),
            dict(
                text=f"Node size: {node_sizing_mode}<br>Edge width: coherence strength",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11)
            )
        ]
    )

    if show_significance:
        fig.add_annotation(
            text="* = Statistically significant module (p<0.01)",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10)
        )

    # Save main figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath',
                                'drawcircle', 'drawrect', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': save_path.stem,
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }

    fig.write_html(str(save_path), config=config)
    print(f"   Saved visualization to: {save_path}")

    # Save additional views if requested
    if save_all_views:
        views_dir = save_path.parent / 'multiple_views'
        views_dir.mkdir(exist_ok=True)

        for view_key in ['anterior', 'posterior', 'left', 'right', 'superior', 'inferior']:
            view_camera = CameraController.get_camera_position(view_key)
            fig.update_layout(
                scene_camera=dict(
                    eye=view_camera['eye'],
                    center=view_camera['center'],
                    up=view_camera['up']
                )
            )
            view_path = views_dir / f"{save_path.stem}_{view_key}.html"
            fig.write_html(str(view_path), config=config)
            print(f"   Saved {view_key} view to: {view_path.name}")

    # Calculate statistics
    stats = {
        'total_nodes': len(active_nodes),
        'total_edges': np.sum(connectivity_matrix != 0) // 2,
        'n_modules': len(unique_modules),
        'node_role_distribution': pd.Series(node_roles).value_counts().to_dict(),
        'Q_total': module_data.get('Q_total', 0),
        'Q_z_score': module_data.get('Q_z_score', 0),
        'camera_view': camera_view
    }

    return fig, stats


def create_interactive_camera_control_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    roi_coords_df: pd.DataFrame,
    connectivity_matrix: np.ndarray,
    module_data: Dict,
    save_dir: Union[str, Path]
) -> None:
    """
    Create an interactive HTML page with camera controls for exploring the visualization.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices
    faces : np.ndarray
        Mesh faces
    roi_coords_df : pd.DataFrame
        ROI coordinates
    connectivity_matrix : np.ndarray
        Connectivity matrix
    module_data : dict
        Module data
    save_dir : str or Path
        Directory to save the panel
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create the interactive HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Modularity Viewer - Interactive Camera Control</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                height: 100vh;
            }
            #controls {
                width: 300px;
                padding: 20px;
                background: #f5f5f5;
                overflow-y: auto;
                border-right: 2px solid #ddd;
            }
            #plot {
                flex-grow: 1;
                height: 100vh;
            }
            .control-group {
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .control-group h3 {
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 5px;
            }
            input[type="number"] {
                width: 60px;
                padding: 5px;
                margin: 2px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            button {
                padding: 8px 15px;
                margin: 5px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                transition: background 0.3s;
            }
            button:hover {
                background: #45a049;
            }
            .preset-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 5px;
            }
            .coordinate-input {
                display: flex;
                align-items: center;
                margin: 5px 0;
            }
            .coordinate-input label {
                width: 30px;
                font-weight: bold;
            }
            #info {
                padding: 10px;
                background: #e8f5e9;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div id="controls">
            <h2 style="color: #333;">Camera Controls</h2>

            <div class="control-group">
                <h3>Preset Views</h3>
                <div class="preset-buttons">
                    <button onclick="setView('anterior')">Anterior</button>
                    <button onclick="setView('posterior')">Posterior</button>
                    <button onclick="setView('left')">Left</button>
                    <button onclick="setView('right')">Right</button>
                    <button onclick="setView('superior')">Superior</button>
                    <button onclick="setView('inferior')">Inferior</button>
                </div>
            </div>

            <div class="control-group">
                <h3>Manual Camera Position</h3>
                <div class="coordinate-input">
                    <label>X:</label>
                    <input type="number" id="cam-x" value="1.5" step="0.1">
                </div>
                <div class="coordinate-input">
                    <label>Y:</label>
                    <input type="number" id="cam-y" value="1.5" step="0.1">
                </div>
                <div class="coordinate-input">
                    <label>Z:</label>
                    <input type="number" id="cam-z" value="1.5" step="0.1">
                </div>
                <button onclick="applyManualCamera()">Apply Position</button>
            </div>

            <div id="info">
                <strong>Tips:</strong><br>
                - Drag to rotate<br>
                - Scroll to zoom<br>
                - Double-click to reset<br>
                - Right-click drag to pan
            </div>
        </div>

        <div id="plot"></div>

        <script>
            const presets = {
                anterior: {eye: {x:0, y:2, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                posterior: {eye: {x:0, y:-2, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                left: {eye: {x:-2, y:0, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                right: {eye: {x:2, y:0, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                superior: {eye: {x:0, y:0, z:2}, center: {x:0, y:0, z:0}, up: {x:0, y:1, z:0}},
                inferior: {eye: {x:0, y:0, z:-2}, center: {x:0, y:0, z:0}, up: {x:0, y:-1, z:0}}
            };

            function setView(viewName) {
                const camera = presets[viewName];
                Plotly.relayout('plot', {'scene.camera': camera});
            }

            function applyManualCamera() {
                const x = parseFloat(document.getElementById('cam-x').value);
                const y = parseFloat(document.getElementById('cam-y').value);
                const z = parseFloat(document.getElementById('cam-z').value);

                const camera = {
                    eye: {x: x, y: y, z: z},
                    center: {x: 0, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1}
                };

                Plotly.relayout('plot', {'scene.camera': camera});
            }

            const plotData = [];
            const plotLayout = {
                scene: {
                    camera: presets.anterior,
                    xaxis: {visible: false},
                    yaxis: {visible: false},
                    zaxis: {visible: false},
                    bgcolor: 'white',
                    aspectmode: 'data'
                },
                margin: {l: 0, r: 0, t: 0, b: 0},
                showlegend: true
            };

            Plotly.newPlot('plot', plotData, plotLayout);
        </script>
    </body>
    </html>
    """

    panel_path = save_dir / "interactive_camera_control.html"
    with open(panel_path, 'w') as f:
        f.write(html_template)

    print(f"Created interactive camera control panel at: {panel_path}")


def run_enhanced_visualization_pipeline(
    matrix_path: Union[str, Path],
    netneurotools_results_dir: Union[str, Path],
    mesh_file: Union[str, Path],
    roi_coords_file: Union[str, Path],
    output_dir: Union[str, Path],
    k_value: int,
    visualization_types: List[str] = None,
    node_sizing_modes: List[str] = None,
    use_thresholding: bool = True,
    n_top_edges: int = 64,
    base_node_size: int = 12,
    max_node_multiplier: float = 2.0,
    show_labels: bool = True,
    show_significance: bool = True,
    state_mapping: Dict[int, int] = None,
    camera_views: List[str] = None,
    enable_interactive_panel: bool = True,
    save_multiple_views: bool = False
) -> Dict:
    """
    Run enhanced visualization pipeline with camera control features.

    Parameters
    ----------
    matrix_path : str or Path
        Path to connectivity matrices (.npy file)
    netneurotools_results_dir : str or Path
        Path to netneurotools results directory
    mesh_file : str or Path
        Path to brain mesh file
    roi_coords_file : str or Path
        Path to ROI coordinates file
    output_dir : str or Path
        Output directory
    k_value : int
        Number of clusters
    visualization_types : list, optional
        Types of visualizations to create
    node_sizing_modes : list, optional
        Node sizing modes
    use_thresholding : bool
        Whether to use edge thresholding
    n_top_edges : int
        Number of top edges to keep
    base_node_size : int
        Base node size
    max_node_multiplier : float
        Maximum size multiplier
    show_labels : bool
        Whether to show labels
    show_significance : bool
        Whether to show significance
    state_mapping : dict, optional
        State index to label mapping
    camera_views : list, optional
        Camera views to generate
    enable_interactive_panel : bool
        Whether to create interactive panel
    save_multiple_views : bool
        Whether to save all standard views

    Returns
    -------
    dict
        Results dictionary
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default state mapping
    if state_mapping is None:
        state_mapping = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}

    # Default settings
    if visualization_types is None:
        visualization_types = ['all', 'intra', 'inter', 'nodes_only', 'significant_only']

    if node_sizing_modes is None:
        node_sizing_modes = ['pc', 'zscore', 'both']

    if camera_views is None:
        camera_views = ['oblique']

    print("="*80)
    print("ENHANCED BRAIN MODULARITY VISUALIZATION PIPELINE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Visualization types: {visualization_types}")
    print(f"Node sizing modes: {node_sizing_modes}")
    print(f"Camera views: {camera_views}")
    print("="*80)

    # Load mesh and ROI coordinates
    print("\n1. Loading brain mesh and ROI data...")
    vertices, faces = load_mesh_file(mesh_file)

    # Load ROI coordinates
    roi_coords_path = Path(roi_coords_file)
    if roi_coords_path.suffix == '.csv':
        roi_coords_df = pd.read_csv(roi_coords_file)
        roi_coords_df = roi_coords_df.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported ROI file format: {roi_coords_path.suffix}")

    print(f"   Loaded {len(roi_coords_df)} ROIs")

    # Load connectivity matrices
    print(f"\n2. Loading connectivity matrices...")
    matrices = np.load(matrix_path)
    print(f"   Loaded matrices: shape {matrices.shape}")

    # Initialize loader and load results
    print(f"\n3. Loading NetNeurotools results...")
    loader = NetNeurotoolsModularityLoader(netneurotools_results_dir)

    try:
        k_results = loader.load_comprehensive_results(k_value)
        print(f"   Loaded results for {len(k_results['states'])} states")
    except Exception as e:
        print(f"   Error loading results: {e}")
        import traceback
        traceback.print_exc()
        return {}

    all_results = {}

    # Define thresholding configurations
    if use_thresholding:
        thresholding_configs = {
            'thresholded': n_top_edges,
            'non_thresholded': None
        }
    else:
        thresholding_configs = {'non_thresholded': None}

    # Create interactive control panel if enabled
    if enable_interactive_panel and len(k_results['states']) > 0:
        print("\n4. Creating interactive camera control panel...")
        first_state = k_results['states'][0]
        if first_state['state_idx'] < matrices.shape[0]:
            demo_matrix = matrices[first_state['state_idx']]
            demo_matrix[demo_matrix < 0] = 0
            create_interactive_camera_control_panel(
                vertices, faces, roi_coords_df,
                demo_matrix, first_state, output_dir
            )

    # Process each state
    for state_data in k_results['states']:
        state_idx = state_data['state_idx']
        actual_state_label = state_mapping.get(state_idx, state_idx)

        print(f"\n{'='*60}")
        print(f"Processing State {state_idx} (displayed as State {actual_state_label})")
        print(f"{'='*60}")

        if state_idx >= matrices.shape[0]:
            print(f"WARNING: State {state_idx} not found in matrices")
            continue

        connectivity_matrix = matrices[state_idx]
        connectivity_matrix[connectivity_matrix < 0] = 0

        print(f"  Q-score: {state_data.get('Q_total', 0):.3f}")
        print(f"  Z-score: {state_data.get('Q_z_score', 0):.2f}")

        for thresh_name, thresh_value in thresholding_configs.items():
            print(f"\n  {thresh_name.upper()} Configuration:")

            if thresh_value is not None:
                processed_matrix = threshold_matrix_top_n(connectivity_matrix, thresh_value)
                n_edges = np.sum(processed_matrix != 0) // 2
                print(f"    Keeping top {thresh_value} edges (actual: {n_edges})")
            else:
                processed_matrix = connectivity_matrix.copy()
                n_edges = np.sum(processed_matrix != 0) // 2
                print(f"    Using all {n_edges} edges")

            for size_mode in node_sizing_modes:
                print(f"      Node sizing: {size_mode}")

                for viz_type in visualization_types:
                    print(f"        Creating {viz_type} visualization...")

                    for camera_view in camera_views:
                        viz_dir = output_dir / thresh_name / f"node_size_{size_mode}" / f"state_{actual_state_label}" / camera_view
                        viz_dir.mkdir(parents=True, exist_ok=True)

                        try:
                            title = f"k={k_value} - {viz_type.replace('_', ' ').title()}"
                            save_path = viz_dir / f"{viz_type}.html"

                            fig, stats = create_enhanced_modularity_visualization(
                                vertices=vertices,
                                faces=faces,
                                roi_coords_df=roi_coords_df,
                                connectivity_matrix=processed_matrix,
                                module_data=state_data,
                                metrics_df=state_data.get('modules_df'),
                                plot_title=title,
                                save_path=str(save_path),
                                visualization_type=viz_type,
                                node_sizing_mode=size_mode,
                                base_node_size=base_node_size,
                                max_node_multiplier=max_node_multiplier,
                                n_top_edges=None,
                                edge_width_range=(1.0, 6.0),
                                mesh_opacity=0.15,
                                show_labels=show_labels,
                                show_significance=show_significance,
                                border_width=6,
                                actual_state_label=actual_state_label,
                                camera_view=camera_view,
                                enable_camera_controls=True,
                                save_all_views=save_multiple_views
                            )

                            key = f"state{actual_state_label}_{thresh_name}_{size_mode}_{viz_type}_{camera_view}"
                            all_results[key] = {
                                'state_idx': state_idx,
                                'state_label': actual_state_label,
                                'threshold': thresh_name,
                                'node_sizing': size_mode,
                                'viz_type': viz_type,
                                'camera_view': camera_view,
                                'n_edges': n_edges,
                                'Q_total': stats['Q_total'],
                                'Q_z_score': stats['Q_z_score'],
                                'stats': stats,
                                'path': str(save_path)
                            }

                        except Exception as e:
                            print(f"          ERROR: {str(e)}")
                            import traceback
                            traceback.print_exc()

    # Create summary
    print(f"\n{'='*80}")
    print("Creating summary report...")
    summary_path = output_dir / "visualization_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("ENHANCED BRAIN MODULARITY VISUALIZATION SUMMARY\n")
        f.write(f"k={k_value} Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total visualizations created: {len(all_results)}\n")
        f.write(f"Camera views: {camera_views}\n\n")

        displayed_states = sorted(set(r['state_label'] for r in all_results.values()))
        for state_label in displayed_states:
            f.write(f"\nState {state_label}:\n")
            f.write("-"*40 + "\n")
            state_results = [r for r in all_results.values() if r['state_label'] == state_label]
            if state_results:
                f.write(f"  Q-score: {state_results[0]['Q_total']:.3f}\n")
                f.write(f"  Z-score: {state_results[0]['Q_z_score']:.2f}\n")
                f.write(f"  Total visualizations: {len(state_results)}\n")

    print(f"Summary saved to: {summary_path}")

    # Save camera presets reference
    camera_ref_path = output_dir / "camera_presets.json"
    with open(camera_ref_path, 'w') as f:
        json.dump(CameraController.PRESET_VIEWS, f, indent=2)
    print(f"Camera presets saved to: {camera_ref_path}")

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"Total visualizations created: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    return all_results
