"""
Brain Connectivity Visualization
================================
Brain connectivity visualization functions with camera controls.
Version 3 features: Camera controls, toggleable edges, scaled edge widths,
                   vector node sizes, node metrics hover, edge-linked node hiding.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import colorsys
from typing import Dict, Tuple, Optional, Union, List

# Import shared modules from package
from .mesh import load_mesh_file
from .camera import CameraController
from .utils import (
    convert_node_size_input,
    convert_node_color_input,
    load_connectivity_input,
    load_node_metrics,
    calculate_edge_width,
    generate_module_colors
)


def create_brain_connectivity_plot(
    vertices,
    faces,
    roi_coords_df,
    connectivity_matrix: Union[np.ndarray, str, pd.DataFrame],
    plot_title: str = "Brain Connectivity Network",
    save_path: str = "brain_connectivity.html",
    node_size: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, List, Dict, str] = 8,
    node_color: Union[str, np.ndarray, pd.Series, pd.DataFrame, List] = 'purple',
    node_border_color: str = 'magenta',
    pos_edge_color: str = 'red',
    neg_edge_color: str = 'blue',
    edge_width: Union[float, Tuple[float, float]] = (1.0, 5.0),
    edge_threshold: float = 0.0,
    mesh_color: str = 'lightgray',
    mesh_opacity: float = 0.15,
    label_font_size: int = 8,
    fast_render: bool = False,
    camera_view: str = 'oblique',
    custom_camera: Optional[Dict] = None,
    enable_camera_controls: bool = True,
    show_only_connected_nodes: bool = True,
    node_metrics: Optional[Union[str, pd.DataFrame]] = None,
    hide_nodes_with_hidden_edges: bool = True,
    export_image: Optional[str] = None,
    image_format: str = 'png',
    image_dpi: int = 300,
    export_show_title: bool = True,
    export_show_legend: bool = True
):
    """
    Create an interactive 3D brain connectivity visualization.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices array of shape (n_vertices, 3)
    faces : numpy.ndarray
        Mesh faces array of shape (n_faces, 3)
    roi_coords_df : pandas.DataFrame
        DataFrame containing ROI coordinates with columns:
        - 'cog_x', 'cog_y', 'cog_z': world coordinates
        - 'roi_name': name of the ROI
    connectivity_matrix : numpy.ndarray, str, or pd.DataFrame
        Connectivity matrix or path to file. Supports:
        - numpy array: Used directly
        - str: Path to file (.npy, .csv, .txt, .mat, .edge)
        - pd.DataFrame: Converted to numpy array
    plot_title : str, optional
        Title for the plot
    save_path : str, optional
        Path where to save the HTML file
    node_size : int, float, array-like, dict, or str, optional
        Size of the ROI nodes. Can be:
        - Scalar (int/float): All nodes same size
        - numpy array/list/Series: Per-node sizes
        - dict: {node_idx: size}
        - str: Path to file (.csv, .txt, .npy, .mat)
    node_color : str, array-like, or file path, optional
        Color of the ROI nodes. Can be:
        - Single color string: All nodes same color (e.g., 'purple', '#FF0000')
        - numpy array of integers: Module assignments (1-indexed), colors auto-generated
        - numpy array of color strings: Per-node colors
        - pandas Series/DataFrame: Colors or module assignments
        - list: Colors or module assignments
        - str: Path to file (.csv, .npy) containing assignments or colors
    node_border_color : str, optional
        Border color of the ROI nodes
    pos_edge_color : str, optional
        Color for positive connections
    neg_edge_color : str, optional
        Color for negative connections
    edge_width : float or tuple, optional
        Edge width specification:
        - float: Fixed width for all edges
        - tuple (min, max): Scale edge widths based on absolute weight
    edge_threshold : float, optional
        Threshold for showing edges (default 0.0 shows all non-zero)
    mesh_color : str, optional
        Color of the brain mesh
    mesh_opacity : float, optional
        Opacity of the brain mesh
    label_font_size : int, optional
        Font size for ROI labels
    fast_render : bool, optional
        If True, uses optimizations for faster rendering
    camera_view : str, optional
        Camera view preset name (default 'oblique')
    custom_camera : dict, optional
        Custom camera position dict with 'eye', 'center', 'up' keys
    enable_camera_controls : bool, optional
        Whether to enable camera view dropdown controls (default True)
    show_only_connected_nodes : bool, optional
        If True (default), only show nodes with at least one edge
    node_metrics : str or pd.DataFrame, optional
        Node metrics for hover display. Rows = nodes, columns = metric names.
        Can be CSV path or DataFrame.
    hide_nodes_with_hidden_edges : bool, optional
        If True (default), nodes with only positive or only negative edges
        will be hidden when those edge types are toggled off in the legend.
    export_image : str, optional
        Path to export static image (e.g., 'output.png'). If None, no image exported.
        Supported formats: png, svg, pdf (requires kaleido package).
        Note: Exported images use fixed 1200x900 dimensions for consistency.
    image_format : str, optional
        Image format if export_image doesn't have extension: 'png', 'svg', 'pdf'
    image_dpi : int, optional
        DPI for exported PNG/JPEG images (default 300 for publication quality).
        Vector formats (SVG, PDF) ignore this setting.
    export_show_title : bool, optional
        Whether to show the title in exported images (default True).
    export_show_legend : bool, optional
        Whether to show the legend in exported images (default True).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    graph_stats : dict
        Dictionary containing graph statistics
    """

    print(f"Creating brain connectivity visualization: {plot_title}")

    # Load connectivity matrix from various input types
    conn_matrix = load_connectivity_input(connectivity_matrix, n_expected_nodes=len(roi_coords_df))
    print(f"Connectivity matrix shape: {conn_matrix.shape}")

    # Get number of nodes
    n_nodes = len(roi_coords_df)

    # Convert node_size to array
    node_sizes = convert_node_size_input(node_size, n_nodes, default_size=8.0)
    print(f"Node sizes: min={node_sizes.min():.1f}, max={node_sizes.max():.1f}")

    # Convert node_color to appropriate format
    node_colors, module_color_map, module_assignments = convert_node_color_input(
        node_color, n_nodes, default_color='purple'
    )
    is_single_color = isinstance(node_colors, str)
    if not is_single_color:
        print(f"Node colors: {len(set(node_colors))} unique colors")

    # Load node metrics if provided
    metrics_df = None
    if node_metrics is not None:
        metrics_df = load_node_metrics(node_metrics, n_expected_nodes=n_nodes)
        print(f"Loaded node metrics with columns: {list(metrics_df.columns)}")

    # Determine edge width scaling
    if isinstance(edge_width, tuple):
        min_edge_width, max_edge_width = edge_width
        scale_edge_width = True
    else:
        min_edge_width = max_edge_width = float(edge_width)
        scale_edge_width = False

    # Create NetworkX graph
    G_all = nx.Graph()

    # Add all nodes with valid coordinates
    valid_nodes = 0
    for idx, row in roi_coords_df.iterrows():
        if not pd.isna(row['cog_x']):
            G_all.add_node(idx,
                          pos=[row['cog_x'], row['cog_y'], row['cog_z']],
                          label=row['roi_name'],
                          x=row['cog_x'],
                          y=row['cog_y'],
                          z=row['cog_z'],
                          size=node_sizes[idx] if idx < len(node_sizes) else 8.0)
            valid_nodes += 1

    print(f"Added {valid_nodes} nodes with valid coordinates")

    # Add edges based on connectivity matrix
    edge_count = 0
    all_weights = []
    for i in range(conn_matrix.shape[0]):
        for j in range(i + 1, conn_matrix.shape[1]):
            weight = conn_matrix[i, j]
            if abs(weight) > edge_threshold and weight != 0:
                if i in G_all.nodes() and j in G_all.nodes():
                    G_all.add_edge(i, j, weight=weight)
                    all_weights.append(weight)
                    edge_count += 1

    all_weights = np.array(all_weights) if all_weights else np.array([0])
    print(f"Added {edge_count} edges above threshold {edge_threshold}")

    # Classify nodes by their edge types
    nodes_with_pos_only = set()
    nodes_with_neg_only = set()
    nodes_with_both = set()

    for node in G_all.nodes():
        has_pos = False
        has_neg = False
        for neighbor in G_all.neighbors(node):
            weight = G_all[node][neighbor]['weight']
            if weight > 0:
                has_pos = True
            elif weight < 0:
                has_neg = True

        if has_pos and has_neg:
            nodes_with_both.add(node)
        elif has_pos:
            nodes_with_pos_only.add(node)
        elif has_neg:
            nodes_with_neg_only.add(node)

    print(f"Node classification: {len(nodes_with_pos_only)} pos-only, "
          f"{len(nodes_with_neg_only)} neg-only, {len(nodes_with_both)} both")

    # Create connected-only graph
    G_connected = G_all.copy()
    isolated_nodes = list(nx.isolates(G_connected))
    G_connected.remove_nodes_from(isolated_nodes)

    # Helper to build hover text for a node
    def build_node_hover(node_idx, label):
        hover_parts = [f"<b>{label}</b>"]
        if metrics_df is not None and node_idx < len(metrics_df):
            row = metrics_df.iloc[node_idx]
            for col in metrics_df.columns:
                if col not in ['roi_name', 'roi_index', 'node_idx']:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        hover_parts.append(f"{col}: {val:.4f}" if isinstance(val, float) else f"{col}: {val}")
                    else:
                        hover_parts.append(f"{col}: {val}")
        return "<br>".join(hover_parts)

    # Prepare edges with variable width and hover information
    def prepare_edges_with_width(G):
        """Prepare edge traces with scaled widths."""
        pos_traces = []  # List of (x, y, z, width, hover) tuples for each positive edge
        neg_traces = []  # List of (x, y, z, width, hover) tuples for each negative edge

        for edge in G.edges(data=True):
            i, j, data = edge
            weight = data['weight']
            node_i = G.nodes[i]
            node_j = G.nodes[j]

            # Calculate edge width
            if scale_edge_width:
                edge_w = calculate_edge_width(weight, all_weights, min_edge_width, max_edge_width)
            else:
                edge_w = min_edge_width

            hover_text = (f"{node_i['label']} <-> {node_j['label']}<br>"
                         f"Strength: {weight:.4f}")

            edge_data = {
                'x': [node_i['x'], node_j['x']],
                'y': [node_i['y'], node_j['y']],
                'z': [node_i['z'], node_j['z']],
                'width': edge_w,
                'hover': hover_text
            }

            if weight > 0:
                pos_traces.append(edge_data)
            else:
                neg_traces.append(edge_data)

        return pos_traces, neg_traces

    pos_edges, neg_edges = prepare_edges_with_width(G_all)

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
        hoverinfo='skip',
        lighting=dict(ambient=0.8) if fast_render else None
    ))

    # Add positive edges - consolidated into single trace with average width for legend
    if pos_edges:
        # Consolidate all positive edges into one trace
        pos_x, pos_y, pos_z, pos_hover = [], [], [], []
        avg_pos_width = np.mean([e['width'] for e in pos_edges])
        for edge in pos_edges:
            pos_x.extend(edge['x'] + [None])
            pos_y.extend(edge['y'] + [None])
            pos_z.extend(edge['z'] + [None])
            pos_hover.extend([edge['hover'], edge['hover'], ''])

        fig.add_trace(go.Scatter3d(
            x=pos_x,
            y=pos_y,
            z=pos_z,
            mode='lines',
            line=dict(color=pos_edge_color, width=avg_pos_width),
            opacity=0.6,
            hoverinfo='text',
            hovertext=pos_hover,
            showlegend=True,
            visible=True,
            name=f'Positive Edges ({len(pos_edges)})',
            legendgroup='pos_edges'
        ))

    # Add negative edges - consolidated into single trace
    if neg_edges:
        neg_x, neg_y, neg_z, neg_hover = [], [], [], []
        avg_neg_width = np.mean([e['width'] for e in neg_edges])
        for edge in neg_edges:
            neg_x.extend(edge['x'] + [None])
            neg_y.extend(edge['y'] + [None])
            neg_z.extend(edge['z'] + [None])
            neg_hover.extend([edge['hover'], edge['hover'], ''])

        fig.add_trace(go.Scatter3d(
            x=neg_x,
            y=neg_y,
            z=neg_z,
            mode='lines',
            line=dict(color=neg_edge_color, width=avg_neg_width),
            opacity=0.6,
            hoverinfo='text',
            hovertext=neg_hover,
            showlegend=True,
            visible=True,
            name=f'Negative Edges ({len(neg_edges)})',
            legendgroup='neg_edges'
        ))

    # Determine which nodes to show based on show_only_connected_nodes
    if show_only_connected_nodes:
        active_nodes = list(G_connected.nodes())
    else:
        active_nodes = list(G_all.nodes())

    # If hide_nodes_with_hidden_edges is True, we need to create separate node traces
    # for nodes linked to positive edges and nodes linked to negative edges.
    # Nodes with BOTH edge types are added to BOTH traces so they hide when
    # BOTH edge types are hidden (they overlap when both are visible, which is fine).
    if hide_nodes_with_hidden_edges and show_only_connected_nodes:
        # Nodes with ONLY positive edges OR nodes with both (linked to pos_edges)
        pos_linked_nodes = [n for n in active_nodes if n in nodes_with_pos_only or n in nodes_with_both]
        # Nodes with ONLY negative edges OR nodes with both (linked to neg_edges)
        neg_linked_nodes = [n for n in active_nodes if n in nodes_with_neg_only or n in nodes_with_both]

        # Add nodes linked to positive edges (includes nodes with both edge types)
        if pos_linked_nodes:
            node_x = [G_all.nodes[n]['x'] for n in pos_linked_nodes]
            node_y = [G_all.nodes[n]['y'] for n in pos_linked_nodes]
            node_z = [G_all.nodes[n]['z'] for n in pos_linked_nodes]
            node_labels = [G_all.nodes[n]['label'] for n in pos_linked_nodes]
            node_sizes_pos = [G_all.nodes[n]['size'] for n in pos_linked_nodes]
            node_hovers = [build_node_hover(n, G_all.nodes[n]['label']) for n in pos_linked_nodes]

            # Get colors for these nodes
            if is_single_color:
                pos_node_colors = node_colors
            else:
                pos_node_colors = [node_colors[n] for n in pos_linked_nodes]

            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=node_sizes_pos,
                    color=pos_node_colors,
                    opacity=0.9,
                    line=dict(color=node_border_color, width=1)
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=label_font_size, color='black', family='Arial'),
                hoverinfo='text',
                hovertext=node_hovers,
                showlegend=False,
                visible=True,
                name='nodes_pos_linked',
                legendgroup='pos_edges'  # Linked to positive edges
            ))

        # Add nodes linked to negative edges (includes nodes with both edge types)
        if neg_linked_nodes:
            node_x = [G_all.nodes[n]['x'] for n in neg_linked_nodes]
            node_y = [G_all.nodes[n]['y'] for n in neg_linked_nodes]
            node_z = [G_all.nodes[n]['z'] for n in neg_linked_nodes]
            node_labels = [G_all.nodes[n]['label'] for n in neg_linked_nodes]
            node_sizes_neg = [G_all.nodes[n]['size'] for n in neg_linked_nodes]
            node_hovers = [build_node_hover(n, G_all.nodes[n]['label']) for n in neg_linked_nodes]

            # Get colors for these nodes
            if is_single_color:
                neg_node_colors = node_colors
            else:
                neg_node_colors = [node_colors[n] for n in neg_linked_nodes]

            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=node_sizes_neg,
                    color=neg_node_colors,
                    opacity=0.9,
                    line=dict(color=node_border_color, width=1)
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=label_font_size, color='black', family='Arial'),
                hoverinfo='text',
                hovertext=node_hovers,
                showlegend=False,
                visible=True,
                name='nodes_neg_linked',
                legendgroup='neg_edges'  # Linked to negative edges
            ))
    else:
        # Simple mode - all nodes in one trace
        node_x = [G_all.nodes[n]['x'] for n in active_nodes]
        node_y = [G_all.nodes[n]['y'] for n in active_nodes]
        node_z = [G_all.nodes[n]['z'] for n in active_nodes]
        node_labels = [G_all.nodes[n]['label'] for n in active_nodes]
        node_sizes_all = [G_all.nodes[n]['size'] for n in active_nodes]
        node_hovers = [build_node_hover(n, G_all.nodes[n]['label']) for n in active_nodes]

        # Get colors for these nodes
        if is_single_color:
            all_node_colors = node_colors
        else:
            all_node_colors = [node_colors[n] for n in active_nodes]

        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes_all,
                color=all_node_colors,
                opacity=0.9,
                line=dict(color=node_border_color, width=1)
            ),
            text=node_labels,
            textposition='top center',
            textfont=dict(size=label_font_size, color='black', family='Arial'),
            hoverinfo='text',
            hovertext=node_hovers,
            showlegend=False,
            visible=True,
            name='nodes_all'
        ))

    # Set camera position
    if custom_camera is not None:
        camera = custom_camera
    else:
        camera = CameraController.get_camera_position(camera_view)

    # Add camera view to title
    if 'name' in camera:
        full_title = f"{plot_title}<br><i>View: {camera['name']}</i>"
    else:
        full_title = plot_title

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

    # Build camera preset buttons if enabled
    updatemenus = []
    active_button_idx = 0  # Default to first button
    if enable_camera_controls:
        camera_buttons = []
        view_keys = [k for k in CameraController.PRESET_VIEWS.keys() if k != 'custom']

        for idx, view_key in enumerate(view_keys):
            view_data = CameraController.PRESET_VIEWS[view_key]
            # Track which button should be active based on initial camera_view
            if view_key == camera_view:
                active_button_idx = idx

            # Update both camera AND title subtitle
            new_title = f"{plot_title}<br><i>View: {view_data['name']}</i>"
            camera_buttons.append(
                dict(
                    args=[{
                        'scene.camera.eye.x': view_data['eye']['x'],
                        'scene.camera.eye.y': view_data['eye']['y'],
                        'scene.camera.eye.z': view_data['eye']['z'],
                        'scene.camera.center.x': view_data['center']['x'],
                        'scene.camera.center.y': view_data['center']['y'],
                        'scene.camera.center.z': view_data['center']['z'],
                        'scene.camera.up.x': view_data['up']['x'],
                        'scene.camera.up.y': view_data['up']['y'],
                        'scene.camera.up.z': view_data['up']['z'],
                        'title.text': new_title
                    }],
                    label=view_data['name'],
                    method='relayout'
                )
            )

        updatemenus = [
            dict(
                type='dropdown',
                showactive=True,
                active=active_button_idx,
                buttons=camera_buttons,
                x=0.01,
                xanchor='left',
                y=0.99,
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=11)
            )
        ]

    # Update layout with camera controls
    fig.update_layout(
        scene=dict(
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
        ),
        width=1200,
        height=900,
        title={
            'text': full_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
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
                text=camera_control_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.50,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)' if camera_control_text else 'rgba(255,255,255,0)',
                bordercolor='black' if camera_control_text else 'rgba(0,0,0,0)',
                borderwidth=1 if camera_control_text else 0,
                borderpad=4
            ),
            dict(
                text=f"V2: Camera controls enabled" if enable_camera_controls else "",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11, color='gray')
            )
        ]
    )

    # Save the figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': save_path.stem,
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }

    if fast_render:
        config['staticPlot'] = False
        config['scrollZoom'] = True

    fig.write_html(save_path, config=config)

    print(f"Saved interactive visualization to: {save_path}")

    # Export static image if requested
    if export_image is not None:
        export_path = Path(export_image)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from path or parameter
        if export_path.suffix:
            fmt = export_path.suffix[1:].lower()  # Remove leading dot
        else:
            fmt = image_format.lower()
            export_path = export_path.with_suffix(f'.{fmt}')

        if fmt not in ['png', 'svg', 'pdf', 'jpeg', 'webp']:
            print(f"Warning: Unsupported format '{fmt}', using PNG")
            fmt = 'png'
            export_path = export_path.with_suffix('.png')

        # Create a deep copy of the figure for export by serializing to dict first
        # This ensures the export figure is completely independent of the original
        fig_dict = fig.to_dict()

        # Remove interactive elements from the layout dict before creating figure
        if 'layout' in fig_dict:
            fig_dict['layout']['updatemenus'] = []  # Remove dropdown menu
            fig_dict['layout']['annotations'] = []  # Remove camera controls text
            fig_dict['layout']['paper_bgcolor'] = 'white'
            fig_dict['layout']['plot_bgcolor'] = 'white'

            # Optionally hide title
            if not export_show_title:
                fig_dict['layout']['title'] = {'text': ''}

            # Optionally hide legend
            if not export_show_legend:
                fig_dict['layout']['showlegend'] = False

        fig_export = go.Figure(fig_dict)

        # Calculate scale factor for DPI (base is 72 DPI)
        # For vector formats (SVG, PDF), use scale=1 since they're resolution-independent
        # For raster formats (PNG, JPEG), cap scale at 4 to avoid memory issues
        if fmt in ['svg', 'pdf']:
            scale = 1.0  # Vector formats don't need DPI scaling
        else:
            scale = min(image_dpi / 72.0, 4.0)  # Cap at 4x to avoid memory issues
            if image_dpi / 72.0 > 4.0:
                print(f"Note: Scale capped at 4x (effective ~288 DPI) to avoid memory issues")

        print(f"Exporting {fmt.upper()} image...")

        # Fixed export dimensions for consistency
        export_width = 1200
        export_height = 900

        try:
            fig_export.write_image(
                str(export_path),
                format=fmt,
                width=export_width,
                height=export_height,
                scale=scale
            )

            # Verify the file was created
            if export_path.exists():
                file_size = export_path.stat().st_size
                print(f"Exported static image to: {export_path}")
                print(f"  Format: {fmt.upper()}, Size: {file_size/1024:.1f} KB")
                if fmt not in ['svg', 'pdf']:
                    print(f"  Dimensions: {int(export_width*scale)}x{int(export_height*scale)} pixels")
            else:
                print(f"ERROR: Export failed - file was not created at {export_path}")

        except Exception as e:
            print(f"ERROR exporting image: {e}")
            print("=" * 60)
            print("Troubleshooting:")
            print("1. Make sure kaleido is installed: pip install -U kaleido")
            print("2. For PDF export, you may also need: pip install -U kaleido[pdf]")
            print("3. Try using a lower image_dpi value if memory issues occur")
            print("=" * 60)

    # Calculate graph statistics
    graph_stats = {
        'total_nodes': G_all.number_of_nodes(),
        'total_edges': G_all.number_of_edges(),
        'connected_nodes': G_connected.number_of_nodes(),
        'isolated_nodes': len(isolated_nodes),
        'positive_edges': len(pos_edges),
        'negative_edges': len(neg_edges),
        'nodes_with_pos_only': len(nodes_with_pos_only),
        'nodes_with_neg_only': len(nodes_with_neg_only),
        'nodes_with_both': len(nodes_with_both),
        'network_density': nx.density(G_all),
        'average_degree': np.mean([d for n, d in G_all.degree()]) if G_all.number_of_nodes() > 0 else 0,
        'edge_width_scaled': scale_edge_width,
        'node_size_is_vector': not isinstance(node_size, (int, float)),
        'node_color_is_vector': not is_single_color,
        'module_color_map': module_color_map
    }

    if G_all.number_of_nodes() > 0:
        degree_dict = dict(G_all.degree())
        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        graph_stats['top_connected_nodes'] = [(G_all.nodes[node_id]['label'], degree)
                                              for node_id, degree in sorted_nodes]

    return fig, graph_stats


def quick_brain_plot(vertices, faces, roi_coords_df, connectivity_matrix,
                     title="Brain Network", save_name="brain_plot.html"):
    """
    Quick plotting function with default parameters.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices
    faces : numpy.ndarray
        Mesh faces
    roi_coords_df : pandas.DataFrame
        ROI coordinates dataframe
    connectivity_matrix : numpy.ndarray
        Connectivity matrix
    title : str, optional
        Plot title
    save_name : str, optional
        Save filename

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure
    stats : dict
        Graph statistics
    """
    return create_brain_connectivity_plot(
        vertices=vertices,
        faces=faces,
        roi_coords_df=roi_coords_df,
        connectivity_matrix=connectivity_matrix,
        plot_title=title,
        save_path=save_name
    )


def create_brain_connectivity_plot_with_modularity(
    vertices,
    faces,
    roi_coords_df,
    connectivity_matrix: Union[np.ndarray, str, pd.DataFrame],
    module_assignments: Union[np.ndarray, pd.Series, pd.DataFrame, List, str],
    plot_title: str = "Brain Connectivity with Modularity",
    save_path: str = "brain_modularity.html",
    Q_score: Optional[float] = None,
    Z_score: Optional[float] = None,
    node_size: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, List, Dict, str] = 8,
    node_border_color: str = 'darkgray',
    edge_color_mode: str = 'module',
    pos_edge_color: str = 'red',
    neg_edge_color: str = 'blue',
    edge_width: Union[float, Tuple[float, float]] = (1.0, 5.0),
    edge_threshold: float = 0.0,
    mesh_color: str = 'lightgray',
    mesh_opacity: float = 0.15,
    label_font_size: int = 8,
    fast_render: bool = False,
    camera_view: str = 'oblique',
    custom_camera: Optional[Dict] = None,
    enable_camera_controls: bool = True,
    show_only_connected_nodes: bool = True,
    node_metrics: Optional[Union[str, pd.DataFrame]] = None,
    hide_nodes_with_hidden_edges: bool = True,
    export_image: Optional[str] = None,
    image_format: str = 'png',
    image_dpi: int = 300,
    export_show_title: bool = True,
    export_show_legend: bool = True
) -> Tuple[go.Figure, Dict]:
    """
    Create brain connectivity visualization with modularity-based node coloring.

    This function colors nodes based on module assignments and includes a
    module legend. It wraps create_brain_connectivity_plot with modularity
    specific features.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices array of shape (n_vertices, 3)
    faces : numpy.ndarray
        Mesh faces array of shape (n_faces, 3)
    roi_coords_df : pandas.DataFrame
        DataFrame containing ROI coordinates with columns:
        - 'cog_x', 'cog_y', 'cog_z': world coordinates
        - 'roi_name': name of the ROI
    connectivity_matrix : numpy.ndarray, str, or pd.DataFrame
        Connectivity matrix or path to file
    module_assignments : numpy.ndarray, pd.Series, pd.DataFrame, list, or str
        Module assignment for each ROI (1-indexed). Can be:
        - numpy array of integers
        - pandas Series/DataFrame with 'module' column
        - list of integers
        - str: Path to CSV file with module assignments
    plot_title : str, optional
        Title for the plot
    save_path : str, optional
        Path where to save the HTML file
    Q_score : float, optional
        Modularity Q score to display in title (if provided)
    Z_score : float, optional
        Modularity Z score to display in title (if provided)
    node_size : int, float, array-like, dict, or str, optional
        Size of the ROI nodes
    node_border_color : str, optional
        Border color of the ROI nodes (default 'darkgray')
    edge_color_mode : str, optional
        How to color edges: 'module' (default) colors edges by the module
        of the source node, 'sign' colors by positive/negative
    pos_edge_color : str, optional
        Color for positive connections (only used when edge_color_mode='sign')
    neg_edge_color : str, optional
        Color for negative connections (only used when edge_color_mode='sign')
    edge_width : float or tuple, optional
        Edge width specification
    edge_threshold : float, optional
        Threshold for showing edges
    mesh_color : str, optional
        Color of the brain mesh
    mesh_opacity : float, optional
        Opacity of the brain mesh
    label_font_size : int, optional
        Font size for ROI labels
    fast_render : bool, optional
        If True, uses optimizations for faster rendering
    camera_view : str, optional
        Camera view preset name
    custom_camera : dict, optional
        Custom camera position
    enable_camera_controls : bool, optional
        Whether to enable camera view dropdown controls
    show_only_connected_nodes : bool, optional
        If True, only show nodes with at least one edge
    node_metrics : str or pd.DataFrame, optional
        Node metrics for hover display
    hide_nodes_with_hidden_edges : bool, optional
        If True, nodes will be hidden when their edge types are toggled off
    export_image : str, optional
        Path to export static image
    image_format : str, optional
        Image format if export_image doesn't have extension
    image_dpi : int, optional
        DPI for exported images
    export_show_title : bool, optional
        Whether to show the title in exported images
    export_show_legend : bool, optional
        Whether to show the legend in exported images

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    graph_stats : dict
        Dictionary containing graph statistics and module information
    """

    print(f"Creating modularity visualization: {plot_title}")

    # Get number of nodes
    n_nodes = len(roi_coords_df)

    # Process module assignments
    module_arr = None

    if isinstance(module_assignments, str):
        # File path
        path = Path(module_assignments)
        if not path.exists():
            raise FileNotFoundError(f"Module assignments file not found: {module_assignments}")
        df = pd.read_csv(module_assignments)
        if 'module' in df.columns:
            module_arr = df['module'].values
        else:
            # Use last column (assuming it's the module column)
            module_arr = df.iloc[:, -1].values
    elif isinstance(module_assignments, pd.DataFrame):
        if 'module' in module_assignments.columns:
            module_arr = module_assignments['module'].values
        else:
            numeric_cols = module_assignments.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                module_arr = module_assignments[numeric_cols[0]].values
            else:
                module_arr = module_assignments.iloc[:, 0].values
    elif isinstance(module_assignments, pd.Series):
        module_arr = module_assignments.values
    elif isinstance(module_assignments, np.ndarray):
        module_arr = module_assignments.flatten()
    elif isinstance(module_assignments, list):
        module_arr = np.array(module_assignments)
    else:
        raise TypeError(
            f"Unsupported module_assignments type: {type(module_assignments)}. "
            f"Expected np.ndarray, pd.Series, pd.DataFrame, list, or file path."
        )

    # Ensure integer type
    module_arr = module_arr.astype(int)

    # Validate length
    if len(module_arr) != n_nodes:
        raise ValueError(
            f"Module assignments length ({len(module_arr)}) does not match "
            f"number of nodes ({n_nodes})"
        )

    # Build title with Q and Z scores if provided
    full_title = plot_title
    score_parts = []
    if Q_score is not None:
        score_parts.append(f"Q={Q_score:.3f}")
    if Z_score is not None:
        score_parts.append(f"Z={Z_score:.2f}")
    if score_parts:
        full_title = f"{plot_title} ({', '.join(score_parts)})"

    # Validate edge_color_mode
    if edge_color_mode not in ['sign', 'module']:
        raise ValueError(f"edge_color_mode must be 'sign' or 'module', got '{edge_color_mode}'")

    # Generate module colors for edge coloring if needed
    unique_modules = np.unique(module_arr)
    module_colors = generate_module_colors(len(unique_modules))
    module_color_map_internal = {m: module_colors[i] for i, m in enumerate(sorted(unique_modules))}

    # Call the main function with module assignments as node_color
    fig, graph_stats = create_brain_connectivity_plot(
        vertices=vertices,
        faces=faces,
        roi_coords_df=roi_coords_df,
        connectivity_matrix=connectivity_matrix,
        plot_title=full_title,
        save_path=save_path,
        node_size=node_size,
        node_color=module_arr,  # Pass module assignments as node colors
        node_border_color=node_border_color,
        pos_edge_color=pos_edge_color,
        neg_edge_color=neg_edge_color,
        edge_width=edge_width,
        edge_threshold=edge_threshold,
        mesh_color=mesh_color,
        mesh_opacity=mesh_opacity,
        label_font_size=label_font_size,
        fast_render=fast_render,
        camera_view=camera_view,
        custom_camera=custom_camera,
        enable_camera_controls=enable_camera_controls,
        show_only_connected_nodes=show_only_connected_nodes,
        node_metrics=node_metrics,
        hide_nodes_with_hidden_edges=hide_nodes_with_hidden_edges,
        export_image=export_image,
        image_format=image_format,
        image_dpi=image_dpi,
        export_show_title=export_show_title,
        export_show_legend=export_show_legend
    )

    # Get the module color map from stats
    module_color_map = graph_stats.get('module_color_map', {})

    # If edge_color_mode='module', replace the edge traces with module-colored edges
    if edge_color_mode == 'module':
        # Load connectivity matrix to rebuild edges
        conn_matrix = load_connectivity_input(connectivity_matrix, n_expected_nodes=n_nodes)

        # Remove existing edge traces (they have 'Positive Edges' or 'Negative Edges' in name)
        new_data = []
        for trace in fig.data:
            trace_name = trace.name if hasattr(trace, 'name') and trace.name else ''
            if 'Edges' not in trace_name:
                new_data.append(trace)
        fig.data = new_data

        # Build module-colored edges
        # Group edges by the module of the source node (lower index)
        for module_id in sorted(unique_modules):
            module_color = module_color_map_internal[module_id]
            edge_x, edge_y, edge_z, edge_hover = [], [], [], []

            for i in range(conn_matrix.shape[0]):
                for j in range(i + 1, conn_matrix.shape[1]):
                    weight = conn_matrix[i, j]
                    if abs(weight) > edge_threshold and weight != 0:
                        # Color edge by the module of the lower-indexed node
                        if module_arr[i] == module_id:
                            try:
                                edge_x.extend([roi_coords_df.loc[i, 'cog_x'],
                                             roi_coords_df.loc[j, 'cog_x'], None])
                                edge_y.extend([roi_coords_df.loc[i, 'cog_y'],
                                             roi_coords_df.loc[j, 'cog_y'], None])
                                edge_z.extend([roi_coords_df.loc[i, 'cog_z'],
                                             roi_coords_df.loc[j, 'cog_z'], None])
                                hover_text = (f"{roi_coords_df.loc[i, 'roi_name']} (M{module_arr[i]}) <-> "
                                            f"{roi_coords_df.loc[j, 'roi_name']} (M{module_arr[j]})<br>"
                                            f"Strength: {weight:.4f}")
                                edge_hover.extend([hover_text, hover_text, ''])
                            except (KeyError, IndexError):
                                continue

            if edge_x:
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color=module_color, width=2.0),
                    opacity=0.6,
                    hoverinfo='text',
                    hovertext=edge_hover,
                    showlegend=False,
                    name=f'Module {module_id} Edges',
                    legendgroup=f'module_{module_id}'
                ))

        # Update stats
        graph_stats['edge_color_mode'] = 'module'
    else:
        graph_stats['edge_color_mode'] = 'sign'

    # Add module legend using dummy scatter traces
    if module_color_map:
        # Create colored legend entries for each module
        for module_id in sorted(module_color_map.keys()):
            n_in_module = np.sum(module_arr == module_id)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=module_color_map[module_id]),
                name=f'Module {module_id} ({n_in_module})',
                showlegend=True,
                legendgroup=f'module_legend_{module_id}'
            ))

    # Add module statistics to graph_stats
    unique_modules = np.unique(module_arr)
    module_sizes = {f'module_{m}': int(np.sum(module_arr == m)) for m in unique_modules}
    graph_stats['module_assignments'] = module_arr
    graph_stats['n_modules'] = len(unique_modules)
    graph_stats['module_sizes'] = module_sizes
    graph_stats['Q_score'] = Q_score
    graph_stats['Z_score'] = Z_score

    # Re-save the figure with the legend
    save_path = Path(save_path)
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': save_path.stem,
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }
    fig.write_html(save_path, config=config)

    return fig, graph_stats
