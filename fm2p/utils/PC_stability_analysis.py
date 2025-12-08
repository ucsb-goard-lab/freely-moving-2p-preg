import os
import numpy as np
import fm2p

def analyze_PC_stability_remapping(data_paths, stages, environments=[1, 2, 3, 4], place_bin_size=2, plot_flag=True):
    """
    Main function to run the complete place cell stability analysis
    
    Parameters:
    -----------
    data_paths : str or list
        Path(s) to preprocessed data
    stages : list
        List of stage names
    environments : list
        Environment indices to analyze [empty1, home2, empty3, home4]
    plot_flag : bool
        Whether to create plots
        
    Returns:
    --------
    stability_results : dict
        Stability analysis results
    """
    
    # Initialize the class
    sr = fm2p.PC_StabilityRemapping(data_paths, stages, environments, plot_flag)
            
    print("Loading preprocessed data...")
    combined_data = sr.load_preprocessed_data()
    print(f"Loaded data keys: {list(combined_data.keys())}")
    
    print("Extracting activity maps from preprocessed data...")
    activity_maps = sr.extract_activity_maps()
    print(f"Activity maps keys: {list(activity_maps.keys())}")
    
    print("Calculating place field stability...")
    stability_results = sr.calculate_place_field_stability(use_reliable_cells=True, convert_to_cm=True, place_bin_size=place_bin_size)
        
    # Debug: Check if results were generated
    print(f"Stability results keys: {list(stability_results.keys())}")
    if 'home_stability' in stability_results:
        print(f"Home stability data: {stability_results['home_stability']}")
    if 'empty_stability' in stability_results:
        print(f"Empty stability data: {stability_results['empty_stability']}")
        
    print("plotting stability results...")
    sr.plot_stability_comparison_simple(convert_to_cm=True, save_path=data_paths)

    sr.debug_plot_emd_histograms(analysis_type='stability', convert_to_cm=False)
    print("plotting remapping results...")
    sr.plot_remapping_comparison_simple(convert_to_cm=True, save_path=data_paths)
    sr.plot_home_stability_activity_maps(save_path=data_paths)

    
    print("Plotting stability and remapping results across stages...")
    # if plot_flag:
    #     sr._plot_StabilityRemapping_results_multistage(convert_to_cm=True)
    
    
    # Save results
    saved_path = sr.save_stability_results(save_path=data_paths)
    print(f"\nResults saved to: {saved_path}")
    
    
    # print("Printing summary...")
    # sr.print_stability_summary(convert_to_cm=True)
    
    return stability_results

if __name__ == "__main__":
    # Example usage
    data_paths = r"F:\2P\pregnancy\2p_data\250701_NSW130_Baseline3"
    stages = ["baseline1"]
    environments = [1, 2, 3, 4]
    
    # Run analysis
    stability_results = analyze_PC_stability_remapping(
        data_paths,
        stages,
        environments=environments,
        plot_flag=True
    )