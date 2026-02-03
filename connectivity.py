"""
Brain Connectivity Visualization
================================
Brain connectivity visualization functions with camera controls.
Version 2 features: Camera controls, toggleable positive/negative edges.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import colorsys
from typing import Dict, Tuple, Optional

# Import shared modules from package
from .mesh import load_mesh_file
from .camera import CameraController


def create_brain_connectivity_plot(
    vertices,
    faces,
    roi_coords_df,
    connectivity_matrix,
    plot_title="Brain Connectivity Network",
    save_path="brain_connectivity.html",
    node_size=8,
    node_color='purple',
    node_border_color='magenta',
    pos_edge_color='red',
    neg_edge_color='blue',
    edge_width=1.5,
    edge_threshold=0.0,
    mesh_color='lightgray',
    mesh_opacity=0.5,
    label_font_size=8,
    fast_render=False,
    camera_view: str = 'oblique',
    custom_camera: Optional[Dict] = None,
    enable_camera_controls: bool = True,
    show_only_connected_nodes: bool = True
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
    connectivity_matrix : numpy.ndarray
        Connectivity matrix of shape (n_rois, n_rois)
    plot_title : str, optional
        Title for the plot
    save_path : str, optional
        Path where to save the HTML file
    node_size : int, optional
        Size of the ROI nodes
    node_color : str, optional
        Color of the ROI nodes
    node_border_color : str, optional
        Border color of the ROI nodes
    pos_edge_color : str, optional
        Color for positive connections
    neg_edge_color : str, optional
        Color for negative connections
    edge_width : float, optional
        Width of the edge lines
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
        Camera view preset name (default 'oblique'). Options include:
        'anterior', 'posterior', 'left', 'right', 'superior', 'inferior',
        'oblique', 'anterolateral_left', 'anterolateral_right', etc.
    custom_camera : dict, optional
        Custom camera position dict with 'eye', 'center', 'up' keys
    enable_camera_controls : bool, optional
        Whether to enable camera view dropdown controls (default True)
    show_only_connected_nodes : bool, optional
        If True (default), only show nodes that have at least one edge.
        If False, show all nodes including isolated ones.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    graph_stats : dict
        Dictionary containing graph statistics
    """

    print(f"Creating brain connectivity visualization: {plot_title}")

    # Create NetworkX graphs
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
                          z=row['cog_z'])
            valid_nodes += 1

    print(f"Added {valid_nodes} nodes with valid coordinates")

    # Add edges based on connectivity matrix
    edge_count = 0
    for i in range(connectivity_matrix.shape[0]):
        for j in range(i + 1, connectivity_matrix.shape[1]):
            weight = connectivity_matrix[i, j]
            if abs(weight) > edge_threshold and weight != 0:
                if i in G_all.nodes() and j in G_all.nodes():
                    G_all.add_edge(i, j, weight=weight)
                    edge_count += 1

    print(f"Added {edge_count} edges above threshold {edge_threshold}")

    # Create connected-only graph
    G_connected = G_all.copy()
    isolated_nodes = list(nx.isolates(G_connected))
    G_connected.remove_nodes_from(isolated_nodes)

    # Prepare edges with hover information
    def prepare_edges_consolidated(G):
        """Prepare consolidated edge traces for better performance."""
        pos_x, pos_y, pos_z = [], [], []
        neg_x, neg_y, neg_z = [], [], []
        pos_hover, neg_hover = [], []

        for edge in G.edges(data=True):
            i, j, data = edge
            weight = data['weight']
            node_i = G.nodes[i]
            node_j = G.nodes[j]

            hover_text = (f"{node_i['label']} <-> {node_j['label']}<br>"
                         f"Strength: {weight:.4f}")

            if weight > 0:
                pos_x.extend([node_i['x'], node_j['x'], None])
                pos_y.extend([node_i['y'], node_j['y'], None])
                pos_z.extend([node_i['z'], node_j['z'], None])
                pos_hover.extend([hover_text, hover_text, ''])
            else:
                neg_x.extend([node_i['x'], node_j['x'], None])
                neg_y.extend([node_i['y'], node_j['y'], None])
                neg_z.extend([node_i['z'], node_j['z'], None])
                neg_hover.extend([hover_text, hover_text, ''])

        return (pos_x, pos_y, pos_z, pos_hover), (neg_x, neg_y, neg_z, neg_hover)

    # Prepare edges (same for all nodes and connected-only since isolated nodes have no edges)
    pos_all, neg_all = prepare_edges_consolidated(G_all)

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

    # Count edges for legend labels
    pos_edge_count_all = len([i for i in range(0, len(pos_all[0]), 3) if pos_all[0][i] is not None]) if pos_all[0] else 0
    neg_edge_count_all = len([i for i in range(0, len(neg_all[0]), 3) if neg_all[0][i] is not None]) if neg_all[0] else 0

    # Add consolidated edge traces with legend entries for toggling
    if pos_all[0]:
        fig.add_trace(go.Scatter3d(
            x=pos_all[0],
            y=pos_all[1],
            z=pos_all[2],
            mode='lines',
            line=dict(color=pos_edge_color, width=edge_width),
            opacity=0.6,
            hoverinfo='text',
            hovertext=pos_all[3],
            showlegend=True,
            visible=True,
            name=f'Positive Edges ({pos_edge_count_all})',
            legendgroup='pos_edges'
        ))

    if neg_all[0]:
        fig.add_trace(go.Scatter3d(
            x=neg_all[0],
            y=neg_all[1],
            z=neg_all[2],
            mode='lines',
            line=dict(color=neg_edge_color, width=edge_width),
            opacity=0.6,
            hoverinfo='text',
            hovertext=neg_all[3],
            showlegend=True,
            visible=True,
            name=f'Negative Edges ({neg_edge_count_all})',
            legendgroup='neg_edges'
        ))

    # Add nodes with labels - All nodes (shown when show_only_connected_nodes=False)
    node_x_all = [G_all.nodes[i]['x'] for i in G_all.nodes()]
    node_y_all = [G_all.nodes[i]['y'] for i in G_all.nodes()]
    node_z_all = [G_all.nodes[i]['z'] for i in G_all.nodes()]
    node_labels_all = [G_all.nodes[i]['label'] for i in G_all.nodes()]

    fig.add_trace(go.Scatter3d(
        x=node_x_all,
        y=node_y_all,
        z=node_z_all,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            opacity=0.9,
            line=dict(color=node_border_color, width=1)
        ),
        text=node_labels_all,
        textposition='top center',
        textfont=dict(size=label_font_size, color='black', family='Arial'),
        hoverinfo='text',
        hovertext=node_labels_all,
        showlegend=False,
        visible=not show_only_connected_nodes,
        name='nodes_all'
    ))

    # Connected nodes only (shown when show_only_connected_nodes=True, which is the default)
    node_x_conn = [G_connected.nodes[i]['x'] for i in G_connected.nodes()]
    node_y_conn = [G_connected.nodes[i]['y'] for i in G_connected.nodes()]
    node_z_conn = [G_connected.nodes[i]['z'] for i in G_connected.nodes()]
    node_labels_conn = [G_connected.nodes[i]['label'] for i in G_connected.nodes()]

    fig.add_trace(go.Scatter3d(
        x=node_x_conn,
        y=node_y_conn,
        z=node_z_conn,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            opacity=0.9,
            line=dict(color=node_border_color, width=1)
        ),
        text=node_labels_conn,
        textposition='top center',
        textfont=dict(size=label_font_size, color='black', family='Arial'),
        hoverinfo='text',
        hovertext=node_labels_conn,
        showlegend=False,
        visible=show_only_connected_nodes,
        name='nodes_conn'
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

    # Calculate graph statistics
    pos_edges_all_count = len([e for e in G_all.edges(data=True) if e[2]['weight'] > 0])
    neg_edges_all_count = len([e for e in G_all.edges(data=True) if e[2]['weight'] < 0])

    graph_stats = {
        'total_nodes': G_all.number_of_nodes(),
        'total_edges': G_all.number_of_edges(),
        'connected_nodes': G_connected.number_of_nodes(),
        'isolated_nodes': len(isolated_nodes),
        'positive_edges': pos_edges_all_count,
        'negative_edges': neg_edges_all_count,
        'network_density': nx.density(G_all),
        'average_degree': np.mean([d for n, d in G_all.degree()]) if G_all.number_of_nodes() > 0 else 0
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


def create_simple_modularity_visualization(
    vertices,
    faces,
    roi_coords_df,
    connectivity_matrix,
    module_assignments,
    plot_title="Modularity Visualization",
    save_path="modularity_viz.html",
    visualization_type="all",
    node_size=6,
    edge_width=1.5,
    edge_color='red',
    mesh_color='lightgray',
    mesh_opacity=0.5,
    show_labels=True,
    label_font_size=8
) -> Tuple[go.Figure, Dict]:
    """
    Create simple modularity-specific brain connectivity visualization.

    This is a simpler version compared to the enhanced modularity visualization
    in the modularity module. Use this for quick visualizations without
    PC classification or advanced camera controls.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices array of shape (n_vertices, 3)
    faces : numpy.ndarray
        Mesh faces array of shape (n_faces, 3)
    roi_coords_df : pandas.DataFrame
        DataFrame containing ROI coordinates
    connectivity_matrix : numpy.ndarray
        Thresholded connectivity matrix
    module_assignments : numpy.ndarray
        Module assignment for each ROI
    plot_title : str
        Title for the plot
    save_path : str
        Path where to save the HTML file
    visualization_type : str
        Type of edges: "all", "intra", "inter", "nodes_only"
    node_size : int
        Size of the ROI nodes
    edge_width : float
        Width of the edge lines
    edge_color : str
        Color of the edges
    mesh_color : str
        Color of the brain mesh
    mesh_opacity : float
        Opacity of the brain mesh
    show_labels : bool
        Whether to show ROI labels
    label_font_size : int
        Font size for ROI labels

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    module_stats : dict
        Dictionary containing module statistics
    """

    print(f"Creating modularity visualization: {plot_title}")

    # Ensure module_assignments is 1D
    if len(module_assignments.shape) > 1:
        module_assignments = module_assignments.flatten()

    # Identify nodes with connections
    node_has_connection = np.any(connectivity_matrix != 0, axis=0) | np.any(connectivity_matrix != 0, axis=1)
    active_nodes = np.where(node_has_connection)[0]

    print(f"Active nodes after thresholding: {len(active_nodes)} out of {len(module_assignments)}")

    # Filter connectivity matrix based on visualization type
    if visualization_type == "nodes_only":
        filtered_matrix = np.zeros_like(connectivity_matrix)
    elif visualization_type == "intra":
        filtered_matrix = connectivity_matrix.copy()
        for i in range(connectivity_matrix.shape[0]):
            for j in range(connectivity_matrix.shape[1]):
                if module_assignments[i] != module_assignments[j]:
                    filtered_matrix[i, j] = 0
    elif visualization_type == "inter":
        filtered_matrix = connectivity_matrix.copy()
        for i in range(connectivity_matrix.shape[0]):
            for j in range(connectivity_matrix.shape[1]):
                if module_assignments[i] == module_assignments[j]:
                    filtered_matrix[i, j] = 0
    else:
        filtered_matrix = connectivity_matrix.copy()

    # Generate colors for modules
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

    # Prepare edges
    if visualization_type != "nodes_only":
        edge_x, edge_y, edge_z = [], [], []
        edge_hover = []

        for i in range(filtered_matrix.shape[0]):
            for j in range(i + 1, filtered_matrix.shape[1]):
                if filtered_matrix[i, j] != 0:
                    if i in active_nodes and j in active_nodes:
                        edge_x.extend([roi_coords_df.loc[i, 'cog_x'],
                                       roi_coords_df.loc[j, 'cog_x'],
                                       None])
                        edge_y.extend([roi_coords_df.loc[i, 'cog_y'],
                                       roi_coords_df.loc[j, 'cog_y'],
                                       None])
                        edge_z.extend([roi_coords_df.loc[i, 'cog_z'],
                                       roi_coords_df.loc[j, 'cog_z'],
                                       None])

                        hover_text = (f"{roi_coords_df.loc[i, 'roi_name']} (M{int(module_assignments[i])}) <-> "
                                     f"{roi_coords_df.loc[j, 'roi_name']} (M{int(module_assignments[j])})<br>"
                                     f"Strength: {filtered_matrix[i, j]:.4f}")
                        edge_hover.extend([hover_text, hover_text, ''])

        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(color=edge_color, width=edge_width),
                opacity=0.6,
                hoverinfo='text',
                hovertext=edge_hover,
                showlegend=False,
                name='Edges'
            ))

    # Add nodes grouped by module
    for module in unique_modules:
        module_nodes = [i for i in active_nodes if module_assignments[i] == module]

        if module_nodes:
            node_x = [roi_coords_df.loc[i, 'cog_x'] for i in module_nodes]
            node_y = [roi_coords_df.loc[i, 'cog_y'] for i in module_nodes]
            node_z = [roi_coords_df.loc[i, 'cog_z'] for i in module_nodes]

            if show_labels:
                node_text = [roi_coords_df.loc[i, 'roi_name'] for i in module_nodes]
            else:
                node_text = [''] * len(module_nodes)

            hover_text = [f"{roi_coords_df.loc[i, 'roi_name']}<br>Module: {int(module)}"
                         for i in module_nodes]

            fig.add_trace(go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    size=node_size,
                    color=module_to_color[module],
                    opacity=0.9,
                    line=dict(color='darkgray', width=0.5)
                ),
                text=node_text,
                textposition='top center',
                textfont=dict(size=label_font_size, color='black'),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=True,
                name=f'Module {int(module)}'
            ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            dragmode='orbit',
            aspectmode='data'
        ),
        width=1200,
        height=900,
        title={
            'text': plot_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        )
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

    fig.write_html(save_path, config=config)
    print(f"Saved interactive visualization to: {save_path}")

    # Calculate module statistics
    module_stats = {
        'total_active_nodes': int(len(active_nodes)),
        'total_modules': int(n_modules),
        'module_sizes': {},
        'total_edges': int(np.sum(filtered_matrix > 0) // 2),
        'intra_module_edges': 0,
        'inter_module_edges': 0
    }

    for module in unique_modules:
        module_nodes = [i for i in active_nodes if module_assignments[i] == module]
        module_stats['module_sizes'][f'module_{int(module)}'] = int(len(module_nodes))

    for i in range(connectivity_matrix.shape[0]):
        for j in range(i + 1, connectivity_matrix.shape[1]):
            if connectivity_matrix[i, j] != 0:
                if module_assignments[i] == module_assignments[j]:
                    module_stats['intra_module_edges'] += 1
                else:
                    module_stats['inter_module_edges'] += 1

    return fig, module_stats
