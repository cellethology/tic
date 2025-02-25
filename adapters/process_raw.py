import os
import spacegm


dataset_root = "/Users/zhangjiahao/Dataset/CODEX/upmc/dataset"
raw_data_root = "/Users/zhangjiahao/Dataset/CODEX/upmc/raw_data"
# Generate cellular graphs from raw inputs
nx_graph_root = os.path.join(dataset_root, "graph")
fig_save_root = os.path.join(dataset_root, "fig")
model_save_root = os.path.join(dataset_root, "model")

region_ids = set([f.split('.')[0] for f in os.listdir(raw_data_root)])
os.makedirs(nx_graph_root, exist_ok=True)
os.makedirs(fig_save_root, exist_ok=True)
os.makedirs(model_save_root, exist_ok=True)

for region_id in region_ids:
    try:
        graph_output = os.path.join(nx_graph_root, "%s.gpkl" % region_id)
        if not os.path.exists(graph_output):
            print("Processing %s" % region_id)
            G = spacegm.construct_graph_for_region(
                region_id,
                cell_coords_file=os.path.join(raw_data_root, "%s.cell_data.csv" % region_id),
                cell_types_file=os.path.join(raw_data_root, "%s.cell_types.csv" % region_id),
                cell_biomarker_expression_file=os.path.join(raw_data_root, "%s.expression.csv" % region_id),
                cell_features_file=os.path.join(raw_data_root, "%s.cell_features.csv" % region_id),
                voronoi_file=os.path.join(raw_data_root, "%s.json" % region_id),
                graph_output=graph_output,
                voronoi_polygon_img_output=None,
                graph_img_output=os.path.join(fig_save_root, "%s_graph.png" % region_id),
                figsize=10)
    except Exception as e:
        print(region_id, e)
        continue