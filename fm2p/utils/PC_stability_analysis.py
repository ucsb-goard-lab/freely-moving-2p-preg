import os
import numpy as np
import fm2p

def analyze_home_stability(data_paths, stages, place_bin_size=2, plot_flag=True):
    """
    Main function to run home cage place field stability analysis (Home2 → Home4)
    
    Parameters:
    -----------
    data_paths : str or list
        Path(s) to preprocessed data
    stages : list
        List of stage names (e.g., ['baseline1', 'baseline2'])
    place_bin_size : float, default 2
        Size of spatial bins in cm
    plot_flag : bool, default True
        Whether to create plots
        
    Returns:
    --------
    stability_results : dict
        Home cage stability analysis results
    """
    
    # Initialize the class
    sr = fm2p.PC_StabilityRemapping(data_paths, stages, plot_flag)
            
    print("Loading preprocessed data...")
    combined_data = sr.load_preprocessed_data()
    print(f"Loaded data keys: {list(combined_data.keys())}")
    
    print("Extracting activity maps from preprocessed data...")
    activity_maps = sr.extract_activity_maps()
    print(f"Activity maps keys: {list(activity_maps.keys())}")
    
    print("Calculating home cage place field stability (Home2 → Home4)...")
    stability_results = sr.calculate_place_field_stability(
        use_reliable_cells=True, 
        convert_to_cm=True, 
        place_bin_size=place_bin_size
    )
    
    # Debug: Check if results were generated
    print(f"Stability results keys: {list(stability_results.keys())}")
    if 'home_stability' in stability_results:
        for stage_idx in stability_results['home_stability'].keys():
            n_cells = len(stability_results['home_stability'][stage_idx])
            print(f"  Stage {stage_idx}: {n_cells} cells analyzed")
    
    if plot_flag:
        print("\nPlotting stability results...")
        sr.plot_home_stability_violin(convert_to_cm=True, save_path=data_paths)
        
        print("\nPlotting home stability histograms...")
        sr.plot_home_stability_histograms(convert_to_cm=True)
        
        print("\nPlotting home stability activity maps...")
        sr.plot_home_stability_activity_maps(place_bin_size=place_bin_size, save_path=data_paths)
    
    # Print summary
    print("\nPrinting stability summary...")
    sr.print_home_stability_summary(convert_to_cm=True)
    
    # Save results
    saved_path = sr.save_stability_results(save_path=data_paths)
    print(f"\nResults saved to: {saved_path}")
    
    return stability_results

def compare_home_stability_across_sessions(session_files, session_names=None, save_path=None):
    """
    Compare home cage stability across multiple recording sessions
    
    This function loads saved .pkl files from individual sessions and creates
    a bar plot comparison with statistical analysis (ANOVA).
    
    Parameters:
    -----------
    session_files : list of str
        Paths to saved stability results .pkl files
        Example:
        [
            r"F:\2P\pregnancy\2p_data\250701_NSW130_Baseline1\home_stability_baseline1_20250116.pkl",
            r"F:\2P\pregnancy\2p_data\250715_NSW130_Baseline2\home_stability_baseline2_20250116.pkl",
            r"F:\2P\pregnancy\2p_data\250801_NSW130_Gestation1\home_stability_gestation1_20250116.pkl"
        ]
    session_names : list of str, optional
        Custom names for each session (e.g., ["Baseline1", "Baseline2", "Gestation#1"])
        If None, uses filenames
    save_path : str, optional
        Directory where comparison plot will be saved
        If None, plot is only displayed
        
    Returns:
    --------
    sessions_data : dict
        Dictionary containing all session data and statistics
    """
    
    print("\n" + "="*70)
    print("COMPARING HOME CAGE STABILITY ACROSS SESSIONS")
    print("="*70)
    
    # Use the static method from PC_StabilityRemapping class
    fm2p.PC_StabilityRemapping.plot_home_stability_by_session(
        session_files=session_files,
        session_names=session_names,
        save_path=save_path,
        convert_to_cm=True
    )
    
    # Load and return session data for further analysis
    sessions_data = fm2p.PC_StabilityRemapping.load_multiple_sessions(
        session_files=session_files,
        session_names=session_names
    )
    
    return sessions_data

if __name__ == "__main__":
    # Example usage for single session analysis
    data_paths = r"F:\2P\pregnancy\2p_data\250701_NSW130_Baseline3"
    stages = ["baseline3"]
    
    # Run analysis
    stability_results = analyze_home_stability(
        data_paths,
        stages,
        place_bin_size=2,
        plot_flag=True
    )