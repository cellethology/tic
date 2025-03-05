"""
ui.py

A simple GUI that configures and runs the unified_pseudotime_pipeline.py script, 
showing progress and allowing the user to view or save plots.
"""

import PySimpleGUI as sg
import subprocess
import os
import sys
import tempfile
import shutil
import torch
import numpy as np

# Suppose these come from your code or constants
DEFAULT_ROOT = "../../data"
DEFAULT_CELLS_INPUT = "../../data/center_cells.pt"
DEFAULT_CELLS_OUTPUT = "../../data/cells_with_pseudotime.pt"
DEFAULT_PSEUDOTIME_PLOT_SAVE = "../../data/pseudotime_vs_biomarker.png"
DEFAULT_EMBEDDING_PLOT_SAVE = "../../data/embedding_by_cluster.png"

def run_pipeline_with_progress(command_list, progress_title="Running..."):
    """
    Runs the pipeline in a subprocess while displaying a progress bar window.
    This is a simplistic approach that just shows a marquee progress until the command finishes.
    
    :param command_list: A list of strings representing the command (e.g. ["python", "unified_pseudotime_pipeline.py", ...])
    :param progress_title: The window title for the progress.
    """
    layout = [
        [sg.Text(progress_title)],
        [sg.ProgressBar(100, orientation='h', size=(40, 20), key='PROG')],
        [sg.Cancel()]
    ]
    window = sg.Window("Progress", layout, finalize=True)
    progress_bar = window['PROG']

    # Start the subprocess
    proc = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # We do a simple loop reading lines from stdout.
    # We'll update the progress bar in some arbitrary increments or upon certain textual cues.
    progress_value = 0
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(line, end="")  # We can optionally print to console
            # Heuristic: if we see "Step A" or "Step B", we can update progress
            if "Step A" in line:
                progress_value = 20
            elif "Step B" in line:
                progress_value = 50
            elif "Step C" in line:
                progress_value = 80
            else:
                progress_value = min(99, progress_value + 1)

        event, _ = window.read(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Cancel':
            proc.terminate()
            break

        progress_bar.UpdateBar(progress_value)

    window.close()

def main():
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Unified Pseudotime Pipeline UI")],
        [sg.Checkbox("Extract Representation", default=False, key="-EXTRACT-")],
        [sg.Checkbox("Perform Pseudotime", default=False, key="-DO_PSEUDO-")],
        [sg.Frame("Paths", [
            [sg.Text("Root Folder:"), sg.Input(DEFAULT_ROOT, key="-ROOT-", size=(40,1))],
            [sg.Text("Cells Input:"), sg.Input(DEFAULT_CELLS_INPUT, key="-IN-", size=(40,1)), sg.FileBrowse()],
            [sg.Text("Cells Output:"), sg.Input(DEFAULT_CELLS_OUTPUT, key="-OUT-", size=(40,1)), sg.FileSaveAs()]
        ])],
        [sg.Frame("Representation / DR / Clustering", [
            [sg.Text("Representation Key:"), sg.Input("raw_expression", key="-REP-", size=(20,1))],
            [sg.Text("DR Method:"), sg.Combo(["PCA","UMAP"], default_value="PCA", key="-DRM-"), 
             sg.Text("n_components:"), sg.Input("2", key="-NC-", size=(5,1))],
            [sg.Text("Cluster Method:"), sg.Combo(["kmeans","agg"], default_value="kmeans", key="-CLM-"), 
             sg.Text("n_clusters:"), sg.Input("5", key="-NCL-", size=(5,1))],
            [sg.Text("Start Node:"), sg.Input("0", key="-START-", size=(5,1)),
             sg.Text("num_cells:"), sg.Input("", key="-NUMC-", size=(8,1))]
        ])],
        [sg.Frame("Plotting", [
            [sg.Checkbox("Plot Embedding", default=False, key="-PLOT_EMB-"), 
             sg.Text("Save Embedding:"), sg.Input(DEFAULT_EMBEDDING_PLOT_SAVE, key="-EMB_OUT-", size=(30,1)), sg.FileSaveAs()],
            [sg.Checkbox("Plot Pseudotime vs. Features", default=False, key="-PLOT_PSEUDO-"),
             sg.Text("Save Plot:"), sg.Input(DEFAULT_PSEUDOTIME_PLOT_SAVE, key="-PT_OUT-", size=(30,1)), sg.FileSaveAs()],
            [sg.Text("Biomarkers (space-sep):"), sg.Input("", key="-BIO-", size=(20,1)),
             sg.Text("Neighbor Types:"), sg.Input("", key="-NEIG-", size=(20,1))],
            [sg.Text("Bins:"), sg.Input("100", key="-BINS-", size=(5,1))]
        ])],
        [sg.Button("Run Pipeline"), sg.Exit()]
    ]

    window = sg.Window("Pseudotime Pipeline UI", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        if event == "Run Pipeline":
            # Gather settings
            extract_flag = values["-EXTRACT-"]
            do_pseudo_flag = values["-DO_PSEUDO-"]

            root_val = values["-ROOT-"]
            in_val = values["-IN-"]
            out_val = values["-OUT-"]
            rep_key = values["-REP-"]

            dr_method = values["-DRM-"]
            n_comp = values["-NC-"]
            cl_method = values["-CLM-"]
            n_clust = values["-NCL-"]
            start_node = values["-START-"]
            num_cells = values["-NUMC-"]

            # Plot settings
            plot_emb = values["-PLOT_EMB-"]
            plot_pseudo = values["-PLOT_PSEUDO-"]
            emb_out = values["-EMB_OUT-"]
            pt_out = values["-PT_OUT-"]
            biomarkers = values["-BIO-"].strip()
            neighbor_types = values["-NEIG-"].strip()
            bins_val = values["-BINS-"]

            # Build the command
            cmd = [sys.executable, "unified_pseudotime_pipeline.py"]
            if extract_flag:
                cmd.append("--extract_representation")
                cmd.extend(["--root", root_val])
            if do_pseudo_flag:
                cmd.append("--do_pseudotime")
                cmd.extend(["--cells_input", in_val])
                cmd.extend(["--cells_output", out_val])
                cmd.extend(["--representation_key", rep_key])
                cmd.extend(["--dr_method", dr_method])
                cmd.extend(["--n_components", n_comp])
                cmd.extend(["--cluster_method", cl_method])
                cmd.extend(["--n_clusters", n_clust])
                cmd.extend(["--start_node", start_node])
                if num_cells != "":
                    cmd.extend(["--num_cells", num_cells])
                cmd.extend(["--output_dir", os.path.join(root_val, "pseudotime_plots")])
            # Plot
            if plot_emb:
                cmd.append("--plot_embedding")
                cmd.extend(["--embedding_plot_save", emb_out])
            if plot_pseudo:
                cmd.append("--plot_pseudotime_features")
                cmd.extend(["--pseudotime_plot_bins", bins_val])
                if biomarkers:
                    # space-sep
                    biom_list = biomarkers.split()
                    cmd.extend(["--plot_biomarkers"]+biom_list)
                elif neighbor_types:
                    neig_list = neighbor_types.split()
                    cmd.extend(["--plot_neighbor_types"]+neig_list)
                if pt_out:
                    cmd.extend(["--pseudotime_plot_save", pt_out])

            # Show progress while pipeline runs
            run_pipeline_with_progress(cmd, progress_title="Running Pseudotime Pipeline...")

    window.close()

if __name__ == "__main__":
    main()