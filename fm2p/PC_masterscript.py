# import matplotlib
# matplotlib.use('Agg')

# from fm2p.utils.deconcatenate_2pdata import deconcatenate_twopdata
# from fm2p.utils.hippocampus_preprocessing import hippocampal_preprocess
# from fm2p.utils.PC_stability_analysis import analyze_home_stability
# import fm2p

# # ============================================
# # SINGLE SESSION ANALYSIS
# # ============================================

# # Configuration 
# data_paths = r"F:\TEST_mini2pdata\251014_JSY_NSW130_G6"
# cfg_path = r'F:\TEST_mini2pdata\251014_JSY_NSW130_G6\config_HP.yaml'
# cfg = fm2p.read_yaml(cfg_path)

# # # Preprocessing steps
# deconcatenate_twopdata(data_paths, cfg_path)
# hippocampal_preprocess(cfg_path)

# # Analysis parameters
# stages = ["G6"]
# place_bin_size = cfg['place_bin_size']
# plot_flag = True

# # Run home cage stability analysis (Home2 → Home4)
# results = analyze_home_stability(
#     data_paths, 
#     stages, 
#     place_bin_size=place_bin_size,
#     plot_flag=plot_flag
# )

# print("\n" + "="*70)
# print("SINGLE SESSION ANALYSIS COMPLETE")
# print("="*70)
# print(f"Results saved to: {data_paths}")
# print("\nTo compare across multiple sessions, see the section below.")


# ============================================
# CROSS-SESSION COMPARISON (Run after collecting multiple sessions)
# ============================================

# Uncomment and modify the section below after you have run the analysis
# on multiple recording sessions and have saved .pkl files for each

from fm2p.utils.PC_stability_analysis import compare_home_stability_across_sessions

# List all saved result files from individual sessions
session_files = [
    r"F:\TEST_mini2pdata\250908_JSY_NSW130_B1\home_stability_B1_20260117_215043.pkl",
    r"F:\TEST_mini2pdata\250912_JSY_NSW130_B2\home_stability_B2_20260117_161747.pkl",
    r"F:\TEST_mini2pdata\250924_JSY_NSW130_B5\home_stability_B5_20260117_223416.pkl",
    r"F:\TEST_mini2pdata\250928_AJZ_NSW130_G2\home_stability_G2_20260117_224215.pkl",
    r"F:\TEST_mini2pdata\251002_JSY_NSW130_G3\home_stability_G3_20260117_230114.pkl",
    r"F:\TEST_mini2pdata\251006_JSY_NSW130_G4\home_stability_G4_20260117_232038.pkl",
    r"F:\TEST_mini2pdata\251010_AJZ_NSW130_G5\home_stability_G5_20260117_160351.pkl",
    r"F:\TEST_mini2pdata\251014_JSY_NSW130_G6\home_stability_G6_20260117_233822.pkl"
]

# session_files = [
#     r"e:\TEST_mini2pdata\baseline1\home_stability_baseline1_20250116_143022.pkl",
#     r"e:\TEST_mini2pdata\baseline2\home_stability_baseline2_20250116_150045.pkl",
#     r"e:\TEST_mini2pdata\gestation1\home_stability_gestation1_20250116_153015.pkl",
#     r"e:\TEST_mini2pdata\gestation2\home_stability_gestation2_20250116_160030.pkl"
# ]
# Custom names for each session (optional - will use filenames if not provided)
session_names = ["Baseline#1","Baseline#2","Baseline#5", "Gestation#2", "Gestation#3", "Gestation#4", "Gestation#5", "Gestation#6"]

# Directory to save comparison plots
save_path = r"F:\TEST_mini2pdata\cross_session_analysis"

# Run cross-session comparison
print("\n" + "="*70)
print("RUNNING CROSS-SESSION COMPARISON")
print("="*70)

comparison_results = compare_home_stability_across_sessions(
    session_files,
    session_names=session_names,
    save_path=save_path
)

print("\nCross-session comparison complete!")
print(f"Results saved to: {save_path}")