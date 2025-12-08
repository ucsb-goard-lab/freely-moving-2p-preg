import matplotlib
matplotlib.use('Agg')

from fm2p.utils.deconcatenate_2pdata import deconcatenate_twopdata
from fm2p.utils.hippocampus_preprocessing import hippocampal_preprocess
from fm2p.utils.PC_stability_analysis import analyze_PC_stability_remapping
import fm2p

# Configuration 
data_paths = r"e:\TEST_mini2pdata\251010_AJZ_NSW130_G5"
cfg_path = r'e:\TEST_mini2pdata\251010_AJZ_NSW130_G5\config_HP.yaml'
cfg = fm2p.read_yaml(cfg_path)

# Preprocessing steps
deconcatenate_twopdata(data_paths, cfg_path)
hippocampal_preprocess(cfg_path)

# Analysis parameters
stages = ["baseline2"]
environments = [1, 2, 3, 4]
plot_flag = True
place_bin_size = cfg['place_bin_size']

# Run stability analysis (activity maps will be loaded automatically from preproc.h5 files)
results = analyze_PC_stability_remapping(
    data_paths, 
    stages, 
    environments, 
    place_bin_size=place_bin_size,
    plot_flag=plot_flag
)


# # ============================================
# # COMPARE ACROSS SESSIONS
# # ============================================

# # Gather all saved result files
# session_files = [
#     r"F:\2P\pregnancy\2p_data\250701_NSW130_Baseline3\stability_results_baseline3_20250121_143022.pkl",
#     r"F:\2P\pregnancy\2p_data\250715_NSW130_Pregnancy_Early\stability_results_pregnancy_early_20250121_150045.pkl",
#     r"F:\2P\pregnancy\2p_data\250801_NSW130_Pregnancy_Late\stability_results_pregnancy_late_20250121_153015.pkl"
# ]

# session_names = ["Baseline", "Pregnancy Early", "Pregnancy Late"]

# # Compare across sessions
# comparison_results = compare_stability_across_sessions(
#     session_files,
#     session_names=session_names,
#     save_path=r"F:\2P\pregnancy\2p_data\cross_session_analysis"
# )