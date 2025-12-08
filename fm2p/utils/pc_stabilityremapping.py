import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import fm2p
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class PC_StabilityRemapping:
    """
    Class for analyzing place cell stability and remapping across different environments.
    
    Attributes:
    -----------
    data_paths : list
        List of paths to preprocessed _preproc.h5 files.
    environments : list
        List of environment indices to analyze (default: [1, 2, 3, 4] for empty1, home2, empty3, home4).
    plot_flag : bool
        Whether to create plots (default: True).
    """
    
    def __init__(self, data_paths, stages, environments=[1, 2, 3, 4], plot_flag=True):
        self.data_paths = data_paths
        self.stages = stages
        self.environments = environments
        self.plot_flag = plot_flag
        self.combined_preproc_data = None
        self.activity_maps_dict = None
        
    def load_preprocessed_data(self, h5key=None):
        """
        Load preprocessed data from multiple _preproc.h5 files
        
        Parameters:
        -----------
        h5key : str, optional
            Specific key to load from h5 files
            
        Returns:
        --------
        combined_data : dict
            Dictionary containing combined data from all recordings
        """
        num_stages = len(self.stages)
        
        combined_data = {}
        # find all subfolders in self.data_paths
        subfolders = fm2p.find('*', self.data_paths, MR=False)
        
        if num_stages == 1:
            # Single stage - find all preproc files in the data path
            preproc_data_paths = fm2p.find('*preproc_withActivityMap*.h5', self.data_paths, MR=False)
            
            for envs, path in enumerate(preproc_data_paths):
                preproc_data = fm2p.read_h5(path)
                combined_data[0, envs] = preproc_data  # stage 0, environment envs
                
        else:
            # Multiple stages
            for stage_idx, stage_path in enumerate(self.data_paths):
                preproc_data_paths = fm2p.find('*preproc.h5', stage_path, MR=True)
                
                for envs, path in enumerate(preproc_data_paths):
                    preproc_data = fm2p.read_h5(path)
                    combined_data[stage_idx, envs] = preproc_data
        
        self.combined_preproc_data = combined_data
        print(f"Loaded data for {len(combined_data)} stage-environment combinations")
        
        return combined_data
    
    def extract_activity_maps(self):
        """
        Extract activity maps from the loaded preprocessed data and standardize spatial dimensions
        within environment pairs using weighted spatial pooling
        
        Environment pairs:
        - Empty environments: env0 (empty1) and env2 (empty3) 
        - Home environments: env1 (home2) and env3 (home4)
        
        Returns:
        --------
        activity_maps_dict : dict
            Dictionary containing standardized activity maps for each environment
        """
        
        if self.combined_preproc_data is None:
            raise ValueError("Must load preprocessed data first using load_preprocessed_data()")
        
        activity_maps_dict = {}
        env_names = ['1', '2', '3', '4']
        
        # For single stage analysis, use stage_idx = 0
        stage_idx = 0
        
        print(f"Available data keys: {list(self.combined_preproc_data.keys())}")
        
        # First pass: Extract all activity maps
        raw_activity_maps = {}
        for env_idx, env_name in enumerate(env_names):
            if (stage_idx, env_idx) in self.combined_preproc_data:
                data = self.combined_preproc_data[stage_idx, env_idx]
                
                if 'activity_maps' in data:
                    activity_maps = data['activity_maps']
                    raw_activity_maps[env_idx] = activity_maps
                    print(f"Extracted {env_name} activity maps: shape {activity_maps.shape}")
                else:
                    print(f"Warning: 'activity_maps' not found in {env_name} data")
                    print(f"Available keys: {list(data.keys())}")
            else:
                print(f"Warning: No data found for stage {stage_idx}, environment {env_idx} ({env_name})")
        
        if not raw_activity_maps:
            print("No activity maps found!")
            return {}
        
        print("\n=== Standardizing Environment Pairs ===")
        
        # Define environment pairs
        env_pairs = [
            (0, 2, "Empty environments (empty1 & empty3)"),  # Even indices
            (1, 3, "Home environments (home2 & home4)")       # Odd indices
        ]
        
        # Process each environment pair separately
        for env1_idx, env2_idx, pair_name in env_pairs:
            print(f"\n{pair_name}:")
            
            # Check if both environments in the pair are available
            if env1_idx not in raw_activity_maps or env2_idx not in raw_activity_maps:
                missing = []
                if env1_idx not in raw_activity_maps:
                    missing.append(f"{env_names[env1_idx]} (env{env1_idx})")
                if env2_idx not in raw_activity_maps:
                    missing.append(f"{env_names[env2_idx]} (env{env2_idx})")
                print(f"  Skipping pair - missing: {', '.join(missing)}")
                continue
            
            # Get spatial shapes for this pair
            map1 = raw_activity_maps[env1_idx]
            map2 = raw_activity_maps[env2_idx]
            shape1 = map1.shape[1:]  # Skip n_cells dimension
            shape2 = map2.shape[1:]  # Skip n_cells dimension
            
            print(f"  {env_names[env1_idx]}: {shape1}")
            print(f"  {env_names[env2_idx]}: {shape2}")
            
            # Find minimum dimensions within this pair
            min_height = min(shape1[0], shape2[0])
            min_width = min(shape1[1], shape2[1])
            target_shape = (min_height, min_width)
            
            print(f"  Target shape for pair: {target_shape}")
            
            # Standardize each environment in the pair
            for env_idx in [env1_idx, env2_idx]:
                current_map = raw_activity_maps[env_idx]
                current_shape = current_map.shape[1:]
                
                if current_shape == target_shape:
                    print(f"    {env_names[env_idx]}: No resizing needed")
                    activity_maps_dict[stage_idx, env_idx] = current_map
                else:
                    print(f"    {env_names[env_idx]}: Applying weighted pooling from {current_shape} to {target_shape}")
                    
                    # Apply weighted spatial pooling
                    pooled_activity = self._weighted_spatial_pooling(
                        current_map, target_shape
                    )
                    
                    activity_maps_dict[stage_idx, env_idx] = pooled_activity
                    print(f"      Final shape: {pooled_activity.shape}")
        
        self.activity_maps_dict = activity_maps_dict
        self.stage_idx = stage_idx
        
        print(f"\nSuccessfully extracted and standardized activity maps for {len(activity_maps_dict)} environments")
        print("Environment pairs now have matching spatial dimensions:")
        
        # Print final summary
        if (stage_idx, 0) in activity_maps_dict and (stage_idx, 2) in activity_maps_dict:
            empty_shape = activity_maps_dict[stage_idx, 0].shape[1:]
            print(f"  Empty environments (empty1 & empty3): {empty_shape}")
        
        if (stage_idx, 1) in activity_maps_dict and (stage_idx, 3) in activity_maps_dict:
            home_shape = activity_maps_dict[stage_idx, 1].shape[1:]
            print(f"  Home environments (home2 & home4): {home_shape}")
        
        return activity_maps_dict

    def _weighted_spatial_pooling(self, activity_map, target_shape):
        """
        Apply weighted spatial pooling to downsample activity maps while preserving spatial information
        
        Parameters:
        -----------
        activity_map : np.ndarray
            Original activity map with shape (n_cells, height, width)
        target_shape : tuple
            Target spatial dimensions (target_height, target_width)
        
        Returns:
        --------
        pooled_map : np.ndarray
            Downsampled activity map with shape (n_cells, target_height, target_width)
        """
        
        n_cells, orig_height, orig_width = activity_map.shape
        target_height, target_width = target_shape
        
        # If already the target shape, return copy
        if (orig_height, orig_width) == target_shape:
            return activity_map.copy()
        
        # Create output array
        pooled_map = np.zeros((n_cells, target_height, target_width))
        
        # Calculate scaling factors
        height_scale = orig_height / target_height
        width_scale = orig_width / target_width
        
        # For each target bin, calculate weighted contribution from original bins
        for target_i in range(target_height):
            for target_j in range(target_width):
                
                # Calculate the range in original coordinates that maps to this target bin
                orig_i_start = target_i * height_scale
                orig_i_end = (target_i + 1) * height_scale
                orig_j_start = target_j * width_scale
                orig_j_end = (target_j + 1) * width_scale
                
                # Get integer bounds
                i_start = int(np.floor(orig_i_start))
                i_end = int(np.ceil(orig_i_end))
                j_start = int(np.floor(orig_j_start))
                j_end = int(np.ceil(orig_j_end))
                
                # Ensure bounds are within original array
                i_start = max(0, i_start)
                i_end = min(orig_height, i_end)
                j_start = max(0, j_start)
                j_end = min(orig_width, j_end)
                
                # Calculate weighted sum for all cells
                total_weight = 0
                weighted_activity = np.zeros(n_cells)
                
                for orig_i in range(i_start, i_end):
                    for orig_j in range(j_start, j_end):
                        
                        # Calculate overlap weight
                        i_overlap_start = max(orig_i_start, orig_i)
                        i_overlap_end = min(orig_i_end, orig_i + 1)
                        j_overlap_start = max(orig_j_start, orig_j)
                        j_overlap_end = min(orig_j_end, orig_j + 1)
                        
                        # Weight is the area of overlap
                        weight = (i_overlap_end - i_overlap_start) * (j_overlap_end - j_overlap_start)
                        
                        if weight > 0:
                            # Add weighted contribution from all cells
                            weighted_activity += weight * activity_map[:, orig_i, orig_j]
                            total_weight += weight
                
                # Normalize by total weight to preserve activity density
                if total_weight > 0:
                    pooled_map[:, target_i, target_j] = weighted_activity / total_weight
        
        return pooled_map
    
    def calculate_place_field_stability(self, use_reliable_cells=True, convert_to_cm=True, place_bin_size=2):
        """
        Calculate place field stability using Earth Mover's Distance via POT library
        Modified for single mouse data - using one-way ANOVA with post-hoc tests

        Parameters:
        -----------
        use_reliable_cells : bool, default True
            Whether to use only reliable place cells
            True: reliable cells (using cohens_d only) in both sessions
            False: reliable cells (using multicriteria - sigSI & sigRel & hasPlaceField) in both sessions
        convert_to_cm : bool, default True
            Whether to convert distances from bins to cm
            
        Returns:
        --------
        stability_results : dict
            Contains stability distances and statistics for all stages
        """
                
        if self.combined_preproc_data is None:
            raise ValueError("Must load preprocessed data first using load_preprocessed_data()")
        
        if self.activity_maps_dict is None:
            print("Activity maps not loaded. Extracting from preprocessed data...")
            self.extract_activity_maps()
        
        # Initialize results storage for ALL STAGES
        results = {
            'empty_stability': {},  # Will be {stage_idx: [distances]}
            'home_stability': {},   # Will be {stage_idx: [distances]}
            'empty_cell_ids': {},   # Will be {stage_idx: [cell_indices]}
            'home_cell_ids': {},    # Will be {stage_idx: [cell_indices]}
            'cross_stage_statistics': {  # NEW: Statistics comparing across stages
                'empty_cage_across_stages': None,
                'home_cage_across_stages': None
            }
        }
        
        print("=== Place Field Stability Analysis (Earth Mover's Distance via POT) ===")
        
        # Get all available stage indices
        available_stages = set([key[0] for key in self.combined_preproc_data.keys()])
        print(f"Found {len(available_stages)} stages: {sorted(available_stages)}")
        
        # Loop through all stages
        for stage_idx in sorted(available_stages):
            print(f"\n{'='*60}")
            print(f"PROCESSING STAGE {stage_idx} ({self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'})")
            print(f"{'='*60}")
            
            # Check if all required environments exist for this stage
            required_envs = [0, 1, 2, 3]  # empty1, home2, empty3, home4
            missing_envs = []
            for env_idx in required_envs:
                if (stage_idx, env_idx) not in self.combined_preproc_data:
                    missing_envs.append(env_idx)
                if (stage_idx, env_idx) not in self.activity_maps_dict:
                    missing_envs.append(f"activity_maps_{env_idx}")
            
            if missing_envs:
                print(f"Skipping stage {stage_idx}: Missing environments/data: {missing_envs}")
                continue
            
            # Initialize stage-specific results
            results['empty_stability'][stage_idx] = []
            results['home_stability'][stage_idx] = []
            results['empty_cell_ids'][stage_idx] = []
            results['home_cell_ids'][stage_idx] = []
            
            # EMPTY CAGE STABILITY (empty1 vs empty3)
            print(f"\n1. Calculating empty cage stability (empty1 vs empty3) for stage {stage_idx}...")
            
            # Get place cell indices for empty environments
            empty1_data = self.combined_preproc_data[stage_idx, 0]  # empty1
            empty3_data = self.combined_preproc_data[stage_idx, 2]  # empty3
            
            if use_reliable_cells:
                # Use intersection of reliable cells from both sessions
                empty1_reliable = empty1_data['place_cell_reliability']
                empty3_reliable = empty3_data['place_cell_reliability'] 
                empty_valid_cells = empty1_reliable & empty3_reliable
            else:
                # Use intersection of place cells from both sessions (using multi-criteria)
                empty1_place_cells = empty1_data['place_cell_inds']
                empty3_place_cells = empty3_data['place_cell_inds']
                empty_valid_cells = empty1_place_cells & empty3_place_cells
            
            empty_cell_indices = np.where(empty_valid_cells)[0]
            
            # Calculate EMD for empty cage
            empty1_activity = self.activity_maps_dict[stage_idx, 0]  # Shape: (nCells, X_bins, Y_bins)
            empty3_activity = self.activity_maps_dict[stage_idx, 2]

            empty_emd = []
            
            for cell_idx in empty_cell_indices:
                # Check if cell index is valid for both activity maps
                if (cell_idx < empty1_activity.shape[0] and 
                    cell_idx < empty3_activity.shape[0]):
                    
                    # Get activity maps for this cell
                    map1 = empty1_activity[cell_idx]
                    map3 = empty3_activity[cell_idx]
                    
                    # Calculate EMD between the two spatial distributions
                    emd = self._calculate_emd_stability_pot(map1, map3, place_bin_size, convert_to_cm)
                    
                    if not np.isnan(emd):
                        empty_emd.append(emd)
            
            # Store results for this stage
            results['empty_stability'][stage_idx] = np.array(empty_emd)
            results['empty_cell_ids'][stage_idx] = empty_cell_indices[:len(empty_emd)]

            
            print(f"   Calculated stability for {len(empty_emd)} cells")
            if len(empty_emd) > 0:
                print(f"   Mean stability - EMD value: {np.mean(empty_emd):.3f}")
            
            # HOME CAGE STABILITY (home2 vs home4)
            print(f"\n2. Calculating home cage stability (home2 vs home4) for stage {stage_idx}...")
            
            # Get place cell indices for home environments  
            home2_data = self.combined_preproc_data[stage_idx, 1]  # home2
            home4_data = self.combined_preproc_data[stage_idx, 3]  # home4
            
            if use_reliable_cells:
                home2_reliable = home2_data['place_cell_reliability']
                home4_reliable = home4_data['place_cell_reliability']
                home_valid_cells = home2_reliable & home4_reliable
            else:
                home2_place_cells = home2_data['place_cell_inds']
                home4_place_cells = home4_data['place_cell_inds']
                home_valid_cells = home2_place_cells & home4_place_cells
            
            home_cell_indices = np.where(home_valid_cells)[0]
            
            # Calculate EMD for home cage
            home2_activity = self.activity_maps_dict[stage_idx, 1]
            home4_activity = self.activity_maps_dict[stage_idx, 3]
            
            home_emd = []
            
            for cell_idx in home_cell_indices:
                # Check if cell index is valid for both activity maps
                if (cell_idx < home2_activity.shape[0] and 
                    cell_idx < home4_activity.shape[0]):
                    
                    # Get activity maps for this cell
                    map2 = home2_activity[cell_idx]
                    map4 = home4_activity[cell_idx]
                    
                    # Calculate EMD between the two spatial distributions
                    emd = self._calculate_emd_stability_pot(map2, map4, place_bin_size, convert_to_cm)
                    
                    if not np.isnan(emd):
                        home_emd.append(emd)
            
            # Store results for this stage
            results['home_stability'][stage_idx] = np.array(home_emd)
            results['home_cell_ids'][stage_idx] = home_cell_indices[:len(home_emd)]

            
            print(f"   Calculated stability for {len(home_emd)} cells")
            if len(home_emd) > 0:
                print(f"   Mean stability - EMD value: {np.mean(home_emd):.3f}")
        
        # CROSS-STAGE STATISTICAL COMPARISON (Modified for single mouse)
        print(f"\n{'='*60}")
        print("CROSS-STAGE STATISTICAL COMPARISON (Single Mouse)")
        print(f"{'='*60}")
        
        stages_with_data = [s for s in available_stages 
                        if (len(results['empty_stability'].get(s, [])) > 0 or 
                            len(results['home_stability'].get(s, [])) > 0)]
        
        if len(stages_with_data) < 2:
            print("Need at least 2 stages for cross-stage comparison")
            results['anova_statistics'] = {
                'empty_cage_anova': None,
                'home_cage_anova': None,
                'empty_pairwise': None,
                'home_pairwise': None
            }
        else:
            results['anova_statistics'] = {}
            
            # EMPTY CAGE STABILITY ACROSS STAGES
            print("\n1. Comparing empty cage stability across stages...")
            
            # Prepare data for ANOVA
            empty_distances_all = []
            empty_stage_labels = []
            
            for stage_idx in stages_with_data:
                if len(results['empty_stability'][stage_idx]) > 0:
                    empty_distances_all.extend(results['empty_stability'][stage_idx])
                    stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                    empty_stage_labels.extend([stage_name] * len(results['empty_stability'][stage_idx]))
            
            if len(set(empty_stage_labels)) >= 2:
                # Create DataFrame for ANOVA
                df_empty = pd.DataFrame({
                    'distance': empty_distances_all,
                    'stage': empty_stage_labels
                })
                
                # Perform one-way ANOVA using OLS (equivalent to scipy f_oneway but more detailed)
                model_empty = ols('distance ~ C(stage)', data=df_empty).fit()
                anova_empty = anova_lm(model_empty, typ=2)
                
                f_stat = anova_empty['F'].iloc[0]
                p_value = anova_empty['PR(>F)'].iloc[0]
                
                print(f"   Empty cage - One-way ANOVA:")
                print(f"   F-statistic: {f_stat:.4f}")
                print(f"   p-value: {p_value:.4f}")
                print(f"   df: {anova_empty['df'].iloc[0]:.0f}, {anova_empty['df'].iloc[1]:.0f}")
                
                # Post-hoc pairwise comparisons (Tukey HSD) if significant
                pairwise_empty = None
                if p_value < 0.05:
                    print("   Performing post-hoc pairwise comparisons (Tukey HSD)...")
                    pairwise_empty = pairwise_tukeyhsd(endog=empty_distances_all, 
                                                    groups=empty_stage_labels, 
                                                    alpha=0.05)
                    print(pairwise_empty)
                
                results['anova_statistics']['empty_cage_anova'] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'df_between': anova_empty['df'].iloc[0],
                    'df_within': anova_empty['df'].iloc[1],
                    'stage_names': list(set(empty_stage_labels)),
                    'n_total': len(empty_distances_all)
                }
                results['anova_statistics']['empty_pairwise'] = pairwise_empty
            
            # HOME CAGE STABILITY ACROSS STAGES  
            print("\n2. Comparing home cage stability across stages...")
            
            # Prepare data for ANOVA
            home_distances_all = []
            home_stage_labels = []
            
            for stage_idx in stages_with_data:
                if len(results['home_stability'][stage_idx]) > 0:
                    home_distances_all.extend(results['home_stability'][stage_idx])
                    stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                    home_stage_labels.extend([stage_name] * len(results['home_stability'][stage_idx]))
            
            if len(set(home_stage_labels)) >= 2:
                # Create DataFrame for ANOVA
                df_home = pd.DataFrame({
                    'distance': home_distances_all,
                    'stage': home_stage_labels
                })
                
                # Perform one-way ANOVA
                model_home = ols('distance ~ C(stage)', data=df_home).fit()
                anova_home = anova_lm(model_home, typ=2)
                
                f_stat = anova_home['F'].iloc[0]
                p_value = anova_home['PR(>F)'].iloc[0]
                
                print(f"   Home cage - One-way ANOVA:")
                print(f"   F-statistic: {f_stat:.4f}")
                print(f"   p-value: {p_value:.4f}")
                print(f"   df: {anova_home['df'].iloc[0]:.0f}, {anova_home['df'].iloc[1]:.0f}")
                
                # Post-hoc pairwise comparisons if significant
                pairwise_home = None
                if p_value < 0.05:
                    print("   Performing post-hoc pairwise comparisons (Tukey HSD)...")
                    pairwise_home = pairwise_tukeyhsd(endog=home_distances_all, 
                                                    groups=home_stage_labels, 
                                                    alpha=0.05)
                    print(pairwise_home)
                
                results['anova_statistics']['home_cage_anova'] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'df_between': anova_home['df'].iloc[0],
                    'df_within': anova_home['df'].iloc[1],
                    'stage_names': list(set(home_stage_labels)),
                    'n_total': len(home_distances_all)
                }
                results['anova_statistics']['home_pairwise'] = pairwise_home
        
        # Store results in class
        self.stability_results = results
        
        return results
        
    def _calculate_emd_stability_pot(self, activity_map1, activity_map2, place_bin_size=2, convert_to_cm=True):
        """
        Calculate Earth Mover's Distance between two activity maps using POT library
        
        Parameters:
        -----------
        activity_map1 : np.ndarray
            First spatial activity map (height, width)
        activity_map2 : np.ndarray  
            Second spatial activity map (height, width)
        place_bin_size : float
            Size of each spatial bin in cm
        convert_to_cm : bool
            Whether to convert result to cm units
            
        Returns:
        --------
        emd_distance : float
            Earth Mover's Distance between the two activity maps
        """
        
        import ot
        
        # Replace NaN with 0 for EMD calculation
        map1_clean = np.nan_to_num(activity_map1, nan=0.0)
        map2_clean = np.nan_to_num(activity_map2, nan=0.0)
        
        # Check if both maps have activity
        if np.sum(map1_clean) <= 0 or np.sum(map2_clean) <= 0:
            return np.nan
        
        # Normalize to make them probability distributions (must sum to 1)
        map1_norm = map1_clean / np.sum(map1_clean)
        map2_norm = map2_clean / np.sum(map2_clean)
        
        # Flatten to 1D distributions for POT
        dist1 = map1_norm.ravel()
        dist2 = map2_norm.ravel()
        
        # Create coordinate matrix for spatial positions
        height, width = map1_clean.shape
        y_coords, x_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flatten coordinate matrices and stack to get (N, 2) position matrix
        positions = np.column_stack([x_coords.ravel(), y_coords.ravel()]).astype(np.float64)
        
        # Calculate cost matrix (Euclidean distances between all spatial positions)
        cost_matrix = ot.dist(positions, positions, metric='euclidean')
        
        try:
            # Calculate Earth Mover's Distance using POT
            # This gives the optimal transport cost (Wasserstein-1 distance)
            emd_distance = ot.emd2(dist1, dist2, cost_matrix)
            
            # Convert to cm if requested
            if convert_to_cm:
                emd_distance = emd_distance * place_bin_size
                
            return emd_distance
            
        except Exception as e:
            print(f"        POT EMD calculation failed: {e}")
            return np.nan
    
    def calculate_place_field_remapping(self, use_reliable_cells=True, convert_to_cm=True, place_bin_size=2):
        """
        Calculate place field remapping using Earth Mover's Distance via POT library
        Comparisons: Empty1→Home2, Home2→Empty3, Empty3→Home4
        """
        
        if self.combined_preproc_data is None:
            raise ValueError("Must load preprocessed data first using load_preprocessed_data()")
        
        if self.activity_maps_dict is None:
            print("Activity maps not loaded. Extracting from preprocessed data...")
            self.extract_activity_maps()
        
        # Initialize results storage for ALL STAGES
        results = {
            'empty_to_home_remapping': {},      # Empty1 → Home2 (EMD values)
            'home_to_empty_remapping': {},      # Home2 → Empty3 (EMD values)
            'empty_to_home2_remapping': {},     # Empty3 → Home4 (EMD values)
            'remapping_cell_ids': {},           # Cell IDs for each comparison
            'anova_statistics': {               # Cross-stage statistics
                'empty_to_home_anova': None,
                'home_to_empty_anova': None,
                'empty_to_home2_anova': None
            }
        }
        
        print("=== Place Field Remapping Analysis (Earth Mover's Distance via POT) ===")
        
        # Get all available stage indices
        available_stages = set([key[0] for key in self.combined_preproc_data.keys()])
        print(f"Found {len(available_stages)} stages: {sorted(available_stages)}")
        
        # Loop through all stages
        for stage_idx in sorted(available_stages):
            print(f"\n{'='*60}")
            print(f"PROCESSING REMAPPING FOR STAGE {stage_idx}")
            print(f"{'='*60}")
            
            # Check if all required environments exist
            required_envs = [0, 1, 2, 3]  # empty1, home2, empty3, home4
            missing_envs = []
            for env_idx in required_envs:
                if (stage_idx, env_idx) not in self.combined_preproc_data:
                    missing_envs.append(env_idx)
                if (stage_idx, env_idx) not in self.activity_maps_dict:
                    missing_envs.append(f"activity_maps_{env_idx}")
            
            if missing_envs:
                print(f"Skipping stage {stage_idx}: Missing environments/data: {missing_envs}")
                continue
            
            # Get place cell data for all environments
            env_data = {}
            for env_idx in range(4):
                env_data[env_idx] = self.combined_preproc_data[stage_idx, env_idx]
            
            # Find cells that are place cells in ALL environments (for remapping analysis)
            if use_reliable_cells:
                all_reliable = env_data[0]['place_cell_reliability']
                for env_idx in range(1, 4):
                    all_reliable = all_reliable & env_data[env_idx]['place_cell_reliability']
                valid_cells = all_reliable
                print(f"   Found {np.sum(valid_cells)} cells that are reliable place cells in ALL environments")
            else:
                all_place_cells = env_data[0]['place_cell_inds']
                for env_idx in range(1, 4):
                    all_place_cells = all_place_cells & env_data[env_idx]['place_cell_inds']
                valid_cells = all_place_cells
                print(f"   Found {np.sum(valid_cells)} cells that are place cells in ALL environments")
            
            valid_cell_indices = np.where(valid_cells)[0]
            
            # Get activity maps for all environments
            activity_maps = {}
            for env_idx in range(4):
                activity_maps[env_idx] = self.activity_maps_dict[stage_idx, env_idx]
            
            # Initialize stage-specific results
            results['empty_to_home_remapping'][stage_idx] = []      # Empty1 → Home2
            results['home_to_empty_remapping'][stage_idx] = []      # Home2 → Empty3
            results['empty_to_home2_remapping'][stage_idx] = []     # Empty3 → Home4
            results['remapping_cell_ids'][stage_idx] = []
            
            # Calculate remapping EMD for each valid cell
            empty_to_home_emd = []    # Empty1 → Home2
            home_to_empty_emd = []    # Home2 → Empty3  
            empty_to_home2_emd = []   # Empty3 → Home4
            
            print(f"   Calculating EMD for remapping across {len(valid_cell_indices)} cells...")
            
            for cell_idx in valid_cell_indices:
                
                # Check if cell index is valid for all activity maps
                if all(cell_idx < activity_maps[env_idx].shape[0] for env_idx in range(4)):
                    
                    # Get activity maps for all environments for this cell
                    maps = {}
                    valid_maps = True
                    
                    for env_idx in range(4):
                        activity_map = activity_maps[env_idx][cell_idx]
                        # Check if map has activity
                        if np.sum(np.nan_to_num(activity_map, nan=0.0)) <= 0:
                            valid_maps = False
                            break
                        maps[env_idx] = activity_map
                    
                    if valid_maps:
                        # Calculate EMD for each transition
                        
                        # Empty1 → Home2 (Enter Home)
                        emd = self._calculate_emd_stability_pot(maps[0], maps[1], place_bin_size, convert_to_cm)
                        if not np.isnan(emd):
                            empty_to_home_emd.append(emd)
                        else:
                            continue  # Skip this cell if any EMD calculation fails
                        
                        # Home2 → Empty3 (Return to Empty)
                        emd = self._calculate_emd_stability_pot(maps[1], maps[2], place_bin_size, convert_to_cm)
                        if not np.isnan(emd):
                            home_to_empty_emd.append(emd)
                        else:
                            # Remove the previously added value to keep arrays same length
                            empty_to_home_emd.pop()
                            continue
                        
                        # Empty3 → Home4 (Return to Home)  
                        emd = self._calculate_emd_stability_pot(maps[2], maps[3], place_bin_size, convert_to_cm)
                        if not np.isnan(emd):
                            empty_to_home2_emd.append(emd)
                            results['remapping_cell_ids'][stage_idx].append(cell_idx)
                        else:
                            # Remove the previously added values to keep arrays same length
                            empty_to_home_emd.pop()
                            home_to_empty_emd.pop()
                            continue
            
            # Store results for this stage
            results['empty_to_home_remapping'][stage_idx] = np.array(empty_to_home_emd)
            results['home_to_empty_remapping'][stage_idx] = np.array(home_to_empty_emd)
            results['empty_to_home2_remapping'][stage_idx] = np.array(empty_to_home2_emd)
            
            print(f"   Calculated remapping for {len(empty_to_home_emd)} cells")
            if len(empty_to_home_emd) > 0:
                unit_str = 'EMD score' if not convert_to_cm else 'EMD score (cm)'
                print(f"   Empty1→Home2 (Enter Home): {np.mean(empty_to_home_emd):.3f} ± {stats.sem(empty_to_home_emd):.3f} {unit_str}")
                print(f"   Home2→Empty3 (Return to Empty): {np.mean(home_to_empty_emd):.3f} ± {stats.sem(home_to_empty_emd):.3f} {unit_str}")
                print(f"   Empty3→Home4 (Return to Home): {np.mean(empty_to_home2_emd):.3f} ± {stats.sem(empty_to_home2_emd):.3f} {unit_str}")
        
        # CROSS-STAGE STATISTICAL COMPARISON (for multiple stages/recordings)
        stages_with_data = [s for s in available_stages 
                        if len(results['empty_to_home_remapping'].get(s, [])) > 0]
        
        if len(stages_with_data) >= 2:
            print(f"\n{'='*60}")
            print("CROSS-STAGE REMAPPING COMPARISON (EMD)")
            print(f"{'='*60}")
            
            # Analyze each remapping type across stages
            remapping_types = [
                ('empty_to_home_remapping', 'Empty1→Home2 (Enter Home)'),
                ('home_to_empty_remapping', 'Home2→Empty3 (Return to Empty)'),
                ('empty_to_home2_remapping', 'Empty3→Home4 (Return to Home)')
            ]
            
            for result_key, description in remapping_types:
                print(f"\n{description} across stages:")
                
                # Prepare data for ANOVA
                all_emd_values = []
                all_stage_labels = []
                
                for stage_idx in stages_with_data:
                    if len(results[result_key][stage_idx]) > 0:
                        all_emd_values.extend(results[result_key][stage_idx])
                        stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                        all_stage_labels.extend([stage_name] * len(results[result_key][stage_idx]))
                
                if len(set(all_stage_labels)) >= 2:
                    # Perform ANOVA
                    df = pd.DataFrame({
                        'emd_score': all_emd_values,
                        'stage': all_stage_labels
                    })
                    
                    model = ols('emd_score ~ C(stage)', data=df).fit()
                    anova_result = anova_lm(model, typ=2)
                    
                    f_stat = anova_result['F'].iloc[0]
                    p_value = anova_result['PR(>F)'].iloc[0]
                    
                    print(f"   F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        print("   Significant difference between stages!")
                    else:
                        print("   No significant difference between stages")
                    
                    # Store results
                    anova_key = result_key.replace('_remapping', '_anova')
                    results['anova_statistics'][anova_key] = {
                        'f_stat': f_stat,
                        'p_value': p_value,
                        'df_between': anova_result['df'].iloc[0],
                        'df_within': anova_result['df'].iloc[1],
                        'description': description,
                        'n_total': len(all_emd_values)
                    }
        
        elif len(stages_with_data) == 1:
            print(f"\nOnly one stage available - no cross-stage comparison possible")
            print("Future recordings will enable cross-stage statistical analysis")
        
        # Store results in class
        self.remapping_results = results
        
        return results

    def plot_stability_comparison_simple(self, convert_to_cm=True, save_path=None):
        """
        Simple plot comparing stability between empty cage and home cage environments
        Shows only violin plot and optionally saves the figure
        
        Parameters:
        -----------
        convert_to_cm : bool, default True
            Whether to use cm units
        save_path : str, optional
            Directory path where the figure will be saved as 'stability_comparison.png'
            If None, figure is only displayed
        """

        if not hasattr(self, 'stability_results'):
            print("No stability results found. Run calculate_place_field_stability() first.")
            return
        
        results = self.stability_results
        unit_str = 'cm' if convert_to_cm else 'bins'
        
        # Get data for current stage (assuming stage 0 for now)
        stage_idx = 0
        
        empty_stability = results['empty_stability'].get(stage_idx, [])
        home_stability = results['home_stability'].get(stage_idx, [])
        
        if len(empty_stability) == 0 and len(home_stability) == 0:
            print("No stability data available for plotting")
            return
        
        # Create figure with only one subplot - VIOLIN PLOT ONLY
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Prepare data for plotting
        plot_data = []
        group_labels = []
        
        if len(empty_stability) > 0:
            plot_data.extend(empty_stability)
            group_labels.extend(['Empty Cage\n(Empty1 → Empty3)'] * len(empty_stability))
        
        if len(home_stability) > 0:
            plot_data.extend(home_stability)
            group_labels.extend(['Home Cage\n(Home2 → Home4)'] * len(home_stability))
        
        if len(plot_data) == 0:
            print("No data to plot")
            return
        
        # Create DataFrame
        df = pd.DataFrame({
            'Stability Distance': plot_data,
            'Environment': group_labels
        })
        
        # Violin plot
        sns.violinplot(data=df, x='Environment', y='Stability Distance', ax=ax)
        ax.set_ylabel(f'Place Field Stability\nEuclidean Distance ({unit_str})')
        ax.set_title('Place Field Stability Comparison')
        ax.set_xlabel('')
        
        # Add sample sizes and statistics
        if len(empty_stability) > 0 and len(home_stability) > 0:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(empty_stability, home_stability, equal_var=False)
            
            # Add p-value to plot
            ax.text(0.5, 0.95, f'p = {p_value:.4f}', 
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Add overall title with interpretation
        if len(empty_stability) > 0 and len(home_stability) > 0:
            interpretation = "Lower values = More stable place fields"
            if np.mean(empty_stability) < np.mean(home_stability):
                better_env = "Empty cage shows more stable place fields"
            else:
                better_env = "Home cage shows more stable place fields"
            
            fig.suptitle(f'Place Field Stability Analysis\n{interpretation} • {better_env}', 
                        fontsize=14, y=1.02)
        else:
            fig.suptitle('Place Field Stability Analysis', fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        # Save figure if directory path provided
        if save_path is not None:
            import os
            # Create the full file path
            filename = "stability_comparison.png"
            full_path = os.path.join(save_path, filename)
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save the figure
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        
        plt.show()
        
        # Print summary for boss
        print("\n" + "="*60)
        print("SUMMARY FOR UPDATE")
        print("="*60)
        
        if len(empty_stability) > 0:
            print(f"Empty Cage Stability (Empty1 → Empty3):")
            print(f"  • {len(empty_stability)} place cells analyzed")
            print(f"  • Mean stability: {np.mean(empty_stability):.3f} ± {stats.sem(empty_stability):.3f} {unit_str}")
        
        if len(home_stability) > 0:
            print(f"Home Cage Stability (Home2 → Home4):")
            print(f"  • {len(home_stability)} place cells analyzed") 
            print(f"  • Mean stability: {np.mean(home_stability):.3f} ± {stats.sem(home_stability):.3f} {unit_str}")
        
        if len(empty_stability) > 0 and len(home_stability) > 0:
            print(f"\nStatistical Comparison:")
            print(f"  • t-test p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  • Result: Significant difference between environments")
            else:
                print(f"  • Result: No significant difference between environments")
            
            # Effect size
            pooled_std = np.sqrt(((len(empty_stability)-1)*np.var(empty_stability, ddof=1) + 
                                (len(home_stability)-1)*np.var(home_stability, ddof=1)) / 
                                (len(empty_stability) + len(home_stability) - 2))
            cohens_d = (np.mean(empty_stability) - np.mean(home_stability)) / pooled_std
            print(f"  • Effect size (Cohen's d): {cohens_d:.3f}")

    def plot_home_stability_activity_maps(self, place_bin_size=2, save_path=None, cells_per_page=6):
        """
        Plot activity maps for home cage place cells comparing Home2 vs Home4
        Shows 2 cells per row (4 plots total per row), 6 cells per page
        Uses raw data without normalization
        
        Parameters:
        -----------
        save_path : str, optional
            Directory path to save the PDF file
        cells_per_page : int, default 6
            Number of cells to show per page
        """
        
        if not hasattr(self, 'stability_results'):
            print("No stability results found. Run calculate_place_field_stability() first.")
            return
        
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        
        # Get data for current stage
        stage_idx = 0
        results = self.stability_results
        
        # Get home cage data
        home_stability = results['home_stability'].get(stage_idx, [])
        home_cell_ids = results['home_cell_ids'].get(stage_idx, [])
        
        if len(home_stability) == 0:
            print("No home cage stability data available")
            return
        
        # Get activity maps
        home2_activity = self.activity_maps_dict[stage_idx, 1]  # Home2
        home4_activity = self.activity_maps_dict[stage_idx, 3]  # Home4
        
        print(f"Plotting activity maps for {len(home_cell_ids)} home cage place cells")
        
        # Set up PDF if save path provided
        if save_path is not None:
            import os
            filename = "home_stability_activity_maps_{}.pdf"
            full_path = os.path.join(save_path, filename.format(fm2p.fmt_now(c=True)))
            os.makedirs(save_path, exist_ok=True)
            pdf = PdfPages(full_path)
            print(f"Saving to: {full_path}")
        else:
            pdf = None
        
        # Process cells in batches of 6 per page
        for batch_start in range(0, len(home_cell_ids), cells_per_page):
            batch_end = min(batch_start + cells_per_page, len(home_cell_ids))
            batch_cells = batch_end - batch_start
            
            # Create figure: 3 rows, 4 columns (2 cells per row) with more spacing
            fig, axes = plt.subplots(3, 4, figsize=(16, 14))  # Increased height from 12 to 14
            
            for i in range(batch_cells):
                cell_idx = home_cell_ids[batch_start + i]
                stability_distance = home_stability[batch_start + i]
                
                # Calculate row and column positions
                row = i // 2  # 2 cells per row
                col_start = (i % 2) * 2  # Each cell takes 2 columns
                
                # Get activity maps for this cell - USE RAW DATA
                home2_map = home2_activity[cell_idx]
                home4_map = home4_activity[cell_idx]
                
                # Only replace NaN with 0 for display, keep original values
                home2_display = np.where(np.isnan(home2_map), 0, home2_map)
                home4_display = np.where(np.isnan(home4_map), 0, home4_map)
                
                # Plot Home2 - NO NORMALIZATION, NO COLORBAR
                axes[row, col_start].imshow(home2_display.T, cmap='viridis')
                # Fixed: Split title into multiple lines and adjust fontsize
                axes[row, col_start].set_title(f'Cell {cell_idx}\nHome Recording#2\nEMD: {stability_distance:.2f}', 
                                            fontsize=9, pad=10)  # Reduced fontsize and added padding
                axes[row, col_start].axis('off')
                
                # Plot Home4 - NO NORMALIZATION, NO COLORBAR
                axes[row, col_start + 1].imshow(home4_display.T, cmap='viridis')
                # Fixed: Split title into multiple lines and adjust fontsize
                axes[row, col_start + 1].set_title(f'Cell {cell_idx}\nHome Recording#4\nEMD: {stability_distance:.2f}', 
                                                fontsize=9, pad=10)  # Reduced fontsize and added padding
                axes[row, col_start + 1].axis('off')
            
            # Hide unused subplots
            for i in range(batch_cells, cells_per_page):
                row = i // 2
                col_start = (i % 2) * 2
                axes[row, col_start].axis('off')
                axes[row, col_start + 1].axis('off')
            
            # Overall title with more spacing
            fig.suptitle(f'Home Cage Place Field Stability (Home2 vs Home4)\n'
                        f'Cells {batch_start+1}-{batch_end} of {len(home_cell_ids)} total', 
                        fontsize=14, y=0.97)  # Moved title higher (was 0.95)
            
            # Adjust spacing between subplots
            plt.tight_layout()
            plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.2)  # Added more vertical space (hspace)
            
            # Save or show
            if pdf is not None:
                pdf.savefig(fig, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close(fig)
        
        # Close PDF
        if pdf is not None:
            pdf.close()
            print(f"Activity maps saved to: {full_path}")
        
        # Print summary statistics
        print(f"\n=== HOME CAGE STABILITY SUMMARY ===")
        print(f"Total cells analyzed: {len(home_stability)}")
        print(f"Mean stability: {np.mean(home_stability):.3f} ± {stats.sem(home_stability):.3f} cm")
        print(f"Median stability: {np.median(home_stability):.3f} cm")
        print(f"Range: {np.min(home_stability):.3f} - {np.max(home_stability):.3f} cm")

    def debug_plot_emd_histograms(self, analysis_type='stability', convert_to_cm=True):
        """
        Debug function to plot EMD histograms for each recording in different colors
        Prints to debug console using matplotlib
        
        Parameters:
        -----------
        analysis_type : str, default 'stability'
            'stability' for stability analysis or 'remapping' for remapping analysis
        convert_to_cm : bool, default True
            Whether to use cm units in labels
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        if analysis_type == 'stability':
            if not hasattr(self, 'stability_results'):
                print("No stability results found. Run calculate_place_field_stability() first.")
                return
            results = self.stability_results
            title_prefix = "Stability EMD"
        
        elif analysis_type == 'remapping':
            if not hasattr(self, 'remapping_results'):
                print("No remapping results found. Run calculate_place_field_remapping() first.")
                return
            results = self.remapping_results
            title_prefix = "Remapping EMD"
        
        else:
            print("analysis_type must be 'stability' or 'remapping'")
            return
        
        unit_str = 'Average EMD score' if convert_to_cm else 'Average EMD score'
        
        # Get all available stages
        available_stages = []
        if analysis_type == 'stability':
            for stage_idx in results['empty_stability'].keys():
                if (len(results['empty_stability'][stage_idx]) > 0 or 
                    len(results['home_stability'][stage_idx]) > 0):
                    available_stages.append(stage_idx)
        
        elif analysis_type == 'remapping':
            for stage_idx in results['empty_to_home_remapping'].keys():
                if len(results['empty_to_home_remapping'][stage_idx]) > 0:
                    available_stages.append(stage_idx)
        
        if not available_stages:
            print(f"No {analysis_type} data available")
            return
        
        print(f"\n{'='*60}")
        print(f"DEBUG: {title_prefix.upper()} HISTOGRAMS BY RECORDING")
        print(f"{'='*60}")
        
        # Define colors for different recordings/environments
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        if analysis_type == 'stability':
            # Create figure with subplots for empty and home cage stability
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # EMPTY CAGE STABILITY
            all_empty_data = []
            all_empty_labels = []
            
            for i, stage_idx in enumerate(available_stages):
                empty_data = results['empty_stability'][stage_idx]
                if len(empty_data) > 0:
                    stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                    color = colors[i % len(colors)]
                    
                    # Plot histogram
                    axes[0].hist(empty_data, bins=20, alpha=0.7, color=color, 
                            label=f'{stage_name} (n={len(empty_data)})', density=True)
                    
                    # Collect for combined stats
                    all_empty_data.extend(empty_data)
                    all_empty_labels.extend([stage_name] * len(empty_data))
                    
                    # Print stats to console
                    print(f"\nEMPTY CAGE - {stage_name}:")
                    print(f"  n = {len(empty_data)}")
                    print(f"  Mean ± SEM: {np.mean(empty_data):.3f} ± {np.std(empty_data)/np.sqrt(len(empty_data)):.3f}")
                    print(f"  Median: {np.median(empty_data):.3f}")
                    print(f"  Range: {np.min(empty_data):.3f} - {np.max(empty_data):.3f}")
                    print(f"  Color: {color}")
                    
                    axes[0].set_title(f'Empty Cage Stability\n{unit_str}, {np.mean(empty_data):.3f}')



            axes[0].legend()
            axes[0].set_xlabel(unit_str)
            axes[0].set_ylabel('Density')
            axes[0].grid(True, alpha=0.3)
            
            # HOME CAGE STABILITY
            all_home_data = []
            all_home_labels = []
            
            for i, stage_idx in enumerate(available_stages):
                home_data = results['home_stability'][stage_idx]
                if len(home_data) > 0:
                    stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                    color = colors[i % len(colors)]
                    
                    # Plot histogram
                    axes[1].hist(home_data, bins=20, alpha=0.7, color=color,
                            label=f'{stage_name} (n={len(home_data)})', density=True)
                    
                    # Collect for combined stats
                    all_home_data.extend(home_data)
                    all_home_labels.extend([stage_name] * len(home_data))
                    
                    # Print stats to console
                    print(f"\nHOME CAGE - {stage_name}:")
                    print(f"  n = {len(home_data)}")
                    print(f"  Mean ± SEM: {np.mean(home_data):.3f} ± {np.std(home_data)/np.sqrt(len(home_data)):.3f}")
                    print(f"  Median: {np.median(home_data):.3f}")
                    print(f"  Range: {np.min(home_data):.3f} - {np.max(home_data):.3f}")
                    print(f"  Color: {color}")
                    
                    axes[1].set_title(f'Home Cage Stability\n{unit_str}, {np.mean(home_data):.3f}')

            
            axes[1].legend()
            axes[1].set_xlabel(unit_str)
            axes[1].set_ylabel('Density')
            axes[1].grid(True, alpha=0.3)
            
            # Overall title
            fig.suptitle(f'{title_prefix} Histograms by Recording', fontsize=16)
        
        elif analysis_type == 'remapping':
            # Create figure with subplots for different remapping transitions
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            transition_keys = ['empty_to_home_remapping', 'home_to_empty_remapping', 'empty_to_home2_remapping']
            transition_names = ['Empty1 → Home2\n(Enter Home)', 'Home2 → Empty3\n(Return to Empty)', 'Empty3 → Home4\n(Return to Home)']
            
            for trans_idx, (key, name) in enumerate(zip(transition_keys, transition_names)):
                axes[trans_idx].set_title(f'{name}\n{unit_str}')
                
                print(f"\n{name.replace(chr(10), ' ')}:")  # Replace newline with space for console
                print("-" * 40)
                
                for i, stage_idx in enumerate(available_stages):
                    transition_data = results[key][stage_idx]
                    if len(transition_data) > 0:
                        stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                        color = colors[i % len(colors)]
                        
                        # Plot histogram
                        axes[trans_idx].hist(transition_data, bins=20, alpha=0.7, color=color,
                                        label=f'{stage_name} (n={len(transition_data)})', density=True)
                        
                        # Print stats to console
                        print(f"  {stage_name}:")
                        print(f"    n = {len(transition_data)}")
                        print(f"    Mean ± SEM: {np.mean(transition_data):.3f} ± {np.std(transition_data)/np.sqrt(len(transition_data)):.3f}")
                        print(f"    Median: {np.median(transition_data):.3f}")
                        print(f"    Range: {np.min(transition_data):.3f} - {np.max(transition_data):.3f}")
                        print(f"    Color: {color}")
                
                axes[trans_idx].legend()
                axes[trans_idx].set_xlabel(unit_str)
                axes[trans_idx].set_ylabel('Density')
                axes[trans_idx].grid(True, alpha=0.3)
            
            # Overall title
            fig.suptitle(f'{title_prefix} Histograms by Recording', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        # Print overall summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Available recordings: {available_stages}")
        print(f"Colors used: {colors[:len(available_stages)]}")
        
        if analysis_type == 'stability':
            if all_empty_data:
                print(f"\nCombined Empty Cage (n={len(all_empty_data)}): {np.mean(all_empty_data):.3f} ± {np.std(all_empty_data)/np.sqrt(len(all_empty_data)):.3f}")
            if all_home_data:
                print(f"Combined Home Cage (n={len(all_home_data)}): {np.mean(all_home_data):.3f} ± {np.std(all_home_data)/np.sqrt(len(all_home_data)):.3f}")
        
        print("\nHistogram plots displayed above ↑")

    # Usage examples:
    # sr.debug_plot_emd_histograms(analysis_type='stability', convert_to_cm=True)
    # sr.debug_plot_emd_histograms(analysis_type='remapping', convert_to_cm=True)

    def plot_remapping_comparison_simple(self, convert_to_cm=True, save_path=None):
        """
        Simple plot comparing remapping between different environment transitions using EMD
        Shows violin plot for Empty1→Home2, Home2→Empty3, Empty3→Home4
        
        Parameters:
        -----------
        convert_to_cm : bool, default True
            Whether to use cm units
        save_path : str, optional
            Directory path where the figure will be saved as 'remapping_comparison.png'
            If None, figure is only displayed
        """
        
        if not hasattr(self, 'remapping_results'):
            print("No remapping results found. Run calculate_place_field_remapping() first.")
            return
        
        results = self.remapping_results
        unit_str = 'Average EMD score' if convert_to_cm else 'Average EMD score'
        
        # Get data for current stage (assuming stage 0 for now)
        stage_idx = 0
        
        empty_to_home = results['empty_to_home_remapping'].get(stage_idx, [])      # Empty1→Home2
        home_to_empty = results['home_to_empty_remapping'].get(stage_idx, [])      # Home2→Empty3
        empty_to_home2 = results['empty_to_home2_remapping'].get(stage_idx, [])    # Empty3→Home4
        
        if len(empty_to_home) == 0:
            print("No remapping data available for plotting")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for plotting
        plot_data = []
        group_labels = []
        
        plot_data.extend(empty_to_home)
        group_labels.extend(['Empty1 → Home2\n(Enter Home)'] * len(empty_to_home))
        
        plot_data.extend(home_to_empty)
        group_labels.extend(['Home2 → Empty3\n(Return to Empty)'] * len(home_to_empty))
        
        plot_data.extend(empty_to_home2)
        group_labels.extend(['Empty3 → Home4\n(Return to Home)'] * len(empty_to_home2))
        
        # Create DataFrame
        df = pd.DataFrame({
            'EMD Score': plot_data,
            'Transition': group_labels
        })
        
        # Left plot: Violin plot
        sns.violinplot(data=df, x='Transition', y='EMD Score', ax=axes[0])
        axes[0].set_ylabel(f'Place Field Remapping\n{unit_str}')
        axes[0].set_title('Place Field Remapping Across Environment Transitions')
        axes[0].set_xlabel('')
        axes[0].tick_params(axis='x', rotation=15)
        
        # Add one-way ANOVA if all three comparisons available
        if len(empty_to_home) > 0 and len(home_to_empty) > 0 and len(empty_to_home2) > 0:
            f_stat, p_value = stats.f_oneway(empty_to_home, home_to_empty, empty_to_home2)
            axes[0].text(0.5, 0.95, f'ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}', 
                        transform=axes[0].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Right plot: Bar plot with error bars
        transitions = ['Empty1 → Home2\n(Enter Home)', 'Home2 → Empty3\n(Return to Empty)', 'Empty3 → Home4\n(Return to Home)']
        means = [np.mean(empty_to_home), np.mean(home_to_empty), np.mean(empty_to_home2)]
        sems = [stats.sem(empty_to_home), stats.sem(home_to_empty), stats.sem(empty_to_home2)]
        sample_sizes = [len(empty_to_home), len(home_to_empty), len(empty_to_home2)]
        colors = ['lightgreen', 'orange', 'lightpink']
        
        bars = axes[1].bar(transitions, means, yerr=sems, capsize=8,
                        alpha=0.8, color=colors,
                        error_kw={'elinewidth': 3, 'capthick': 3})
        
        axes[1].set_ylabel(f'Mean Remapping\n{unit_str}')
        axes[1].set_title('Mean Place Field Remapping by Transition')
        axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=15)
        
        # Add sample sizes and values to bars
        for i, (bar, mean_val, sem_val, n) in enumerate(zip(bars, means, sems, sample_sizes)):
            height = bar.get_height()
            
            # Add sample size above error bar
            axes[1].text(bar.get_x() + bar.get_width()/2., height + sem_val + max(means)*0.05,
                        f'n = {n}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Add mean value on the bar
            axes[1].text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{mean_val:.3f}', ha='center', va='center', 
                        fontweight='bold', fontsize=12, color='black')
        
        # Add grid for easier reading
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Overall formatting
        plt.tight_layout()
        
        # Add interpretation
        fig.suptitle(f'Place Field Remapping Analysis (EMD)\nHigher EMD scores = More remapping between environments', 
                    fontsize=14, y=1.05)
        
        # Save figure if directory path provided
        if save_path is not None:
            import os
            filename = "remapping_comparison.png"
            full_path = os.path.join(save_path, filename)
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        
        plt.show()
        
        # Print summary for boss
        print("\n" + "="*70)
        print("REMAPPING SUMMARY FOR UPDATE (EMD)")
        print("="*70)
        
        print(f"Same {len(empty_to_home)} place cells tracked across all 4 environments")
        print(f"\nRemapping Results (EMD Scores):")
        print(f"  Empty1 → Home2 (Enter Home):     {np.mean(empty_to_home):.3f} ± {stats.sem(empty_to_home):.3f} {unit_str}")
        print(f"  Home2 → Empty3 (Return to Empty): {np.mean(home_to_empty):.3f} ± {stats.sem(home_to_empty):.3f} {unit_str}")
        print(f"  Empty3 → Home4 (Return to Home):  {np.mean(empty_to_home2):.3f} ± {stats.sem(empty_to_home2):.3f} {unit_str}")
        
        if len(empty_to_home) > 0 and len(home_to_empty) > 0 and len(empty_to_home2) > 0:
            f_stat, p_value = stats.f_oneway(empty_to_home, home_to_empty, empty_to_home2)
            print(f"\nStatistical Comparison (One-way ANOVA):")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  Result: Significant difference between transition types")
            else:
                print(f"  Result: No significant difference between transition types")
        
        print(f"\nInterpretation:")
        print(f"  • Lower EMD = Less remapping (similar spatial patterns)")
        print(f"  • Higher EMD = More remapping (different spatial patterns)")

    def plot_remapping_activity_maps(self, save_path=None, cells_per_page=4, transition='home_to_empty'):
        """
        Plot activity maps showing remapping across environment transitions
        Shows all 4 environments for each cell (Empty1→Home2→Empty3→Home4)
        
        Parameters:
        -----------
        save_path : str, optional
            Directory path to save the PDF file
        cells_per_page : int, default 4
            Number of cells to show per page
        transition : str, default 'home_to_empty'
            Which transition to focus on for sorting:
            'empty_to_home': Sort by Empty1→Home2 EMD
            'home_to_empty': Sort by Home2→Empty3 EMD  
            'empty_to_home2': Sort by Empty3→Home4 EMD
        """
        
        if not hasattr(self, 'remapping_results'):
            print("No remapping results found. Run calculate_place_field_remapping() first.")
            return
        
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        
        # Get data for current stage
        stage_idx = 0
        results = self.remapping_results
        
        # Get remapping data
        remapping_cell_ids = results['remapping_cell_ids'].get(stage_idx, [])
        empty_to_home_emd = results['empty_to_home_remapping'].get(stage_idx, [])
        home_to_empty_emd = results['home_to_empty_remapping'].get(stage_idx, [])
        empty_to_home2_emd = results['empty_to_home2_remapping'].get(stage_idx, [])
        
        if len(remapping_cell_ids) == 0:
            print("No remapping data available")
            return
        
        # Get activity maps for all environments
        empty1_activity = self.activity_maps_dict[stage_idx, 0]  # Empty1
        home2_activity = self.activity_maps_dict[stage_idx, 1]   # Home2
        empty3_activity = self.activity_maps_dict[stage_idx, 2]  # Empty3
        home4_activity = self.activity_maps_dict[stage_idx, 3]   # Home4
        
        print(f"Plotting activity maps for {len(remapping_cell_ids)} remapping cells")
        
        # Sort cells by the specified transition EMD
        if transition == 'empty_to_home':
            sort_values = empty_to_home_emd
            transition_name = "Empty1→Home2"
        elif transition == 'home_to_empty':
            sort_values = home_to_empty_emd
            transition_name = "Home2→Empty3"
        elif transition == 'empty_to_home2':
            sort_values = empty_to_home2_emd
            transition_name = "Empty3→Home4"
        else:
            sort_values = home_to_empty_emd
            transition_name = "Home2→Empty3"
        
        # Sort by EMD values (highest remapping first)
        sorted_indices = np.argsort(sort_values)[::-1]
        
        # Set up PDF if save path provided
        if save_path is not None:
            import os
            filename = f"remapping_activity_maps_{transition}.pdf"
            full_path = os.path.join(save_path, filename)
            os.makedirs(save_path, exist_ok=True)
            pdf = PdfPages(full_path)
            print(f"Saving to: {full_path}")
        else:
            pdf = None
        
        # Process cells in batches per page
        for batch_start in range(0, len(sorted_indices), cells_per_page):
            batch_end = min(batch_start + cells_per_page, len(sorted_indices))
            batch_cells = batch_end - batch_start
            
            # Create figure: batch_cells rows, 4 columns (one for each environment)
            fig, axes = plt.subplots(batch_cells, 4, figsize=(16, 4*batch_cells))
            
            # Handle case where there's only one cell
            if batch_cells == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(batch_cells):
                idx = sorted_indices[batch_start + i]
                cell_idx = remapping_cell_ids[idx]
                
                # Get EMD scores for this cell
                e_to_h_emd = empty_to_home_emd[idx]
                h_to_e_emd = home_to_empty_emd[idx]
                e_to_h2_emd = empty_to_home2_emd[idx]
                
                # Get activity maps for this cell
                empty1_map = empty1_activity[cell_idx]
                home2_map = home2_activity[cell_idx]
                empty3_map = empty3_activity[cell_idx]
                home4_map = home4_activity[cell_idx]
                
                # Replace NaN with 0 for display
                maps_display = [
                    np.where(np.isnan(empty1_map), 0, empty1_map),
                    np.where(np.isnan(home2_map), 0, home2_map),
                    np.where(np.isnan(empty3_map), 0, empty3_map),
                    np.where(np.isnan(home4_map), 0, home4_map)
                ]
                
                env_names = ['Empty1', 'Home2', 'Empty3', 'Home4']
                emd_values = ['-', f'{e_to_h_emd:.2f}', f'{h_to_e_emd:.2f}', f'{e_to_h2_emd:.2f}']
                
                # Plot all 4 environments
                for env_idx in range(4):
                    axes[i, env_idx].imshow(maps_display[env_idx].T, cmap='viridis')
                    axes[i, env_idx].set_title(f'{env_names[env_idx]}\nEMD: {emd_values[env_idx]}', 
                                            fontsize=10)
                    axes[i, env_idx].axis('off')
                
                # Add cell ID on the left
                axes[i, 0].text(-0.15, 0.5, f'Cell {cell_idx}', 
                            transform=axes[i, 0].transAxes, rotation=90, 
                            va='center', ha='center', fontsize=12, fontweight='bold')
            
            # Overall title
            fig.suptitle(f'Place Field Remapping Across Environments\n'
                        f'Sorted by {transition_name} EMD (Highest First) • '
                        f'Cells {batch_start+1}-{batch_end} of {len(remapping_cell_ids)} total', 
                        fontsize=14, y=0.95)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            # Save or show
            if pdf is not None:
                pdf.savefig(fig, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close(fig)
        
        # Close PDF
        if pdf is not None:
            pdf.close()
            print(f"Activity maps saved to: {full_path}")
        
        # Print summary statistics
        print(f"\n=== REMAPPING SUMMARY ===")
        print(f"Total cells analyzed: {len(remapping_cell_ids)}")
        print(f"Empty1→Home2 EMD: {np.mean(empty_to_home_emd):.3f} ± {stats.sem(empty_to_home_emd):.3f}")
        print(f"Home2→Empty3 EMD: {np.mean(home_to_empty_emd):.3f} ± {stats.sem(home_to_empty_emd):.3f}")
        print(f"Empty3→Home4 EMD: {np.mean(empty_to_home2_emd):.3f} ± {stats.sem(empty_to_home2_emd):.3f}")
        
    
    def _plot_StabilityRemapping_results_multistage(self, convert_to_cm=True):
        """
        Plot both stability and remapping analysis results comparing across stages
        Shows: Home Cage Stability (Home2→Home4) and Home→Empty Remapping (Home2→Empty3)
        """
        unit_str = 'cm' if convert_to_cm else 'bins'
        
        # Check if we have both stability and remapping results
        has_stability = hasattr(self, 'stability_results') and self.stability_results is not None
        has_remapping = hasattr(self, 'remapping_results') and self.remapping_results is not None
        
        if not has_stability and not has_remapping:
            print("No stability or remapping results found for plotting")
            return
        
        # Get all stages with data
        stages_with_data = set()
        
        if has_stability:
            stability_stages = [s for s in self.stability_results['home_stability'].keys() 
                            if len(self.stability_results['home_stability'][s]) > 0]
            stages_with_data.update(stability_stages)
        
        if has_remapping:
            remapping_stages = [s for s in self.remapping_results['home_to_empty_remapping'].keys() 
                            if len(self.remapping_results['home_to_empty_remapping'][s]) > 0]
            stages_with_data.update(remapping_stages)
        
        stages_with_data = sorted(list(stages_with_data))
        
        if not stages_with_data:
            print("No data available for plotting")
            return
        
        # Create figure comparing stages
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # HOME CAGE STABILITY (Home2 → Home4) ACROSS STAGES
        if has_stability:
            print("Plotting home cage stability...")
            stability_plot_data = []
            stability_stage_labels = []
            stability_stage_means = []
            stability_stage_sems = []
            stability_stage_names = []
            
            for stage_idx in stages_with_data:
                if (stage_idx in self.stability_results['home_stability'] and 
                    len(self.stability_results['home_stability'][stage_idx]) > 0):
                    
                    stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                    data = self.stability_results['home_stability'][stage_idx]
                    
                    stability_plot_data.extend(data)
                    stability_stage_labels.extend([stage_name] * len(data))
                    stability_stage_means.append(np.mean(data))
                    stability_stage_sems.append(stats.sem(data))
                    stability_stage_names.append(stage_name)
            
            if len(stability_plot_data) > 0:
                # Violin plot for stability
                df_stability = pd.DataFrame({
                    'Distance': stability_plot_data,
                    'Stage': stability_stage_labels
                })
                
                sns.violinplot(data=df_stability, x='Stage', y='Distance', ax=axes[0, 0])
                axes[0, 0].set_ylabel(f'Stability Distance ({unit_str})')
                axes[0, 0].set_title('Home Cage Stability (Home2 → Home4)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add ANOVA statistics if available
                if ('anova_statistics' in self.stability_results and 
                    self.stability_results['anova_statistics'].get('home_cage_anova') is not None):
                    anova_stats = self.stability_results['anova_statistics']['home_cage_anova']
                    f_stat = anova_stats['f_stat']
                    p_val = anova_stats['p_value']
                    df1 = anova_stats['df_between']
                    df2 = anova_stats['df_within']
                    
                    stats_text = f'F({df1:.0f},{df2:.0f}) = {f_stat:.2f}, p = {p_val:.4f}'
                    axes[0, 0].text(0.5, 0.95, stats_text, 
                                transform=axes[0, 0].transAxes, ha='center', va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                
                # Bar plot of stability means
                bars = axes[0, 1].bar(stability_stage_names, stability_stage_means,
                                    yerr=stability_stage_sems, capsize=5,
                                    alpha=0.7, color='lightcoral',
                                    error_kw={'elinewidth': 2, 'capthick': 2})
                axes[0, 1].set_ylabel(f'Mean Stability Distance ({unit_str})')
                axes[0, 1].set_title('Home Cage Stability - Mean ± SEM')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Add sample sizes to stability bars
                stability_n_cells = []
                for stage_idx in stages_with_data:
                    if (stage_idx in self.stability_results['home_stability'] and 
                        len(self.stability_results['home_stability'][stage_idx]) > 0):
                        stability_n_cells.append(len(self.stability_results['home_stability'][stage_idx]))
                
                for i, (bar, n_cells) in enumerate(zip(bars, stability_n_cells)):
                    if i < len(stability_stage_sems):
                        height = bar.get_height()
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., 
                                    height + stability_stage_sems[i],
                                    f'n={n_cells}', ha='center', va='bottom', fontsize=10)
        else:
            axes[0, 0].text(0.5, 0.5, 'No stability data available', 
                        transform=axes[0, 0].transAxes, ha='center', va='center')
            axes[0, 1].text(0.5, 0.5, 'No stability data available', 
                        transform=axes[0, 1].transAxes, ha='center', va='center')
        
        # HOME TO EMPTY REMAPPING (Home2 → Empty3) ACROSS STAGES  
        if has_remapping:
            print("Plotting home to empty remapping...")
            remapping_plot_data = []
            remapping_stage_labels = []
            remapping_stage_means = []
            remapping_stage_sems = []
            remapping_stage_names = []
            
            for stage_idx in stages_with_data:
                if (stage_idx in self.remapping_results['home_to_empty_remapping'] and 
                    len(self.remapping_results['home_to_empty_remapping'][stage_idx]) > 0):
                    
                    stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                    data = self.remapping_results['home_to_empty_remapping'][stage_idx]
                    
                    remapping_plot_data.extend(data)
                    remapping_stage_labels.extend([stage_name] * len(data))
                    remapping_stage_means.append(np.mean(data))
                    remapping_stage_sems.append(stats.sem(data))
                    remapping_stage_names.append(stage_name)
            
            if len(remapping_plot_data) > 0:
                # Violin plot for remapping
                df_remapping = pd.DataFrame({
                    'Distance': remapping_plot_data,
                    'Stage': remapping_stage_labels
                })
                
                sns.violinplot(data=df_remapping, x='Stage', y='Distance', ax=axes[1, 0])
                axes[1, 0].set_ylabel(f'Remapping Distance ({unit_str})')
                axes[1, 0].set_title('Home → Empty Remapping (Home2 → Empty3)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Add ANOVA statistics if available
                if ('anova_statistics' in self.remapping_results and 
                    self.remapping_results['anova_statistics'].get('home_to_empty_anova') is not None):
                    anova_stats = self.remapping_results['anova_statistics']['home_to_empty_anova']
                    f_stat = anova_stats['f_stat']
                    p_val = anova_stats['p_value']
                    df1 = anova_stats['df_between']
                    df2 = anova_stats['df_within']
                    
                    stats_text = f'F({df1:.0f},{df2:.0f}) = {f_stat:.2f}, p = {p_val:.4f}'
                    axes[1, 0].text(0.5, 0.95, stats_text, 
                                transform=axes[1, 0].transAxes, ha='center', va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
                
                # Bar plot of remapping means
                bars = axes[1, 1].bar(remapping_stage_names, remapping_stage_means,
                                    yerr=remapping_stage_sems, capsize=5,
                                    alpha=0.7, color='lightgreen',
                                    error_kw={'elinewidth': 2, 'capthick': 2})
                axes[1, 1].set_ylabel(f'Mean Remapping Distance ({unit_str})')
                axes[1, 1].set_title('Home → Empty Remapping - Mean ± SEM')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                # Add sample sizes to remapping bars
                remapping_n_cells = []
                for stage_idx in stages_with_data:
                    if (stage_idx in self.remapping_results['home_to_empty_remapping'] and 
                        len(self.remapping_results['home_to_empty_remapping'][stage_idx]) > 0):
                        remapping_n_cells.append(len(self.remapping_results['home_to_empty_remapping'][stage_idx]))
                
                for i, (bar, n_cells) in enumerate(zip(bars, remapping_n_cells)):
                    if i < len(remapping_stage_sems):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., 
                                    height + remapping_stage_sems[i],
                                    f'n={n_cells}', ha='center', va='bottom', fontsize=10)
        else:
            axes[1, 0].text(0.5, 0.5, 'No remapping data available', 
                        transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, 'No remapping data available', 
                        transform=axes[1, 1].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.suptitle('Place Field Stability vs Remapping Comparison Across Stages', 
                    y=1.02, fontsize=16)
        plt.show()
    
    def print_stability_summary(self, convert_to_cm=True):
        """
        Print summary of both stability and remapping results
        Shows: Home Cage Stability (Home2→Home4) and Home→Empty Remapping (Home2→Empty3)
        """
        unit_str = 'cm' if convert_to_cm else 'bins'
        
        # Check what results we have
        has_stability = hasattr(self, 'stability_results') and self.stability_results is not None
        has_remapping = hasattr(self, 'remapping_results') and self.remapping_results is not None
        
        if not has_stability and not has_remapping:
            print("No stability or remapping results found.")
            return
        
        print("\n" + "="*80)
        print("PLACE FIELD ANALYSIS SUMMARY - STABILITY vs REMAPPING")
        print("="*80)
        print("Stability: How much place fields move in SAME environment (Home2 → Home4)")
        print("Remapping: How much place fields move between DIFFERENT environments (Home2 → Empty3)")
        print("="*80)
        
        # Get all stages with any data
        all_stages = set()
        if has_stability:
            all_stages.update(self.stability_results['home_stability'].keys())
        if has_remapping:
            all_stages.update(self.remapping_results['home_to_empty_remapping'].keys())
        
        # Summary for each stage
        for stage_idx in sorted(all_stages):
            stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
            
            print(f"\n{stage_name.upper()} (Stage {stage_idx}):")
            print("-" * 50)
            
            # Home Cage Stability
            if (has_stability and 
                stage_idx in self.stability_results['home_stability'] and 
                len(self.stability_results['home_stability'][stage_idx]) > 0):
                
                stability_data = self.stability_results['home_stability'][stage_idx]
                print(f"  HOME CAGE STABILITY (Home2 → Home4):")
                print(f"    n = {len(stability_data)} cells")
                print(f"    Mean ± SEM: {np.mean(stability_data):.3f} ± {stats.sem(stability_data):.3f} {unit_str}")
                print(f"    Median: {np.median(stability_data):.3f} {unit_str}")
                print(f"    Range: {np.min(stability_data):.3f} - {np.max(stability_data):.3f} {unit_str}")
            else:
                print(f"  HOME CAGE STABILITY: No data available")
            
            # Home to Empty Remapping
            if (has_remapping and 
                stage_idx in self.remapping_results['home_to_empty_remapping'] and 
                len(self.remapping_results['home_to_empty_remapping'][stage_idx]) > 0):
                
                remapping_data = self.remapping_results['home_to_empty_remapping'][stage_idx]
                print(f"  HOME → EMPTY REMAPPING (Home2 → Empty3):")
                print(f"    n = {len(remapping_data)} cells")
                print(f"    Mean ± SEM: {np.mean(remapping_data):.3f} ± {stats.sem(remapping_data):.3f} {unit_str}")
                print(f"    Median: {np.median(remapping_data):.3f} {unit_str}")
                print(f"    Range: {np.min(remapping_data):.3f} - {np.max(remapping_data):.3f} {unit_str}")
            else:
                print(f"  HOME → EMPTY REMAPPING: No data available")
            
            # Direct comparison within stage
            if (has_stability and has_remapping and 
                stage_idx in self.stability_results['home_stability'] and 
                stage_idx in self.remapping_results['home_to_empty_remapping'] and
                len(self.stability_results['home_stability'][stage_idx]) > 0 and
                len(self.remapping_results['home_to_empty_remapping'][stage_idx]) > 0):
                
                stability_mean = np.mean(self.stability_results['home_stability'][stage_idx])
                remapping_mean = np.mean(self.remapping_results['home_to_empty_remapping'][stage_idx])
                
                print(f"  WITHIN-STAGE COMPARISON:")
                if remapping_mean > stability_mean:
                    ratio = remapping_mean / stability_mean
                    print(f"    Remapping > Stability (Remapping is {ratio:.1f}x larger)")
                    print(f"    → Place fields move MORE when changing environments")
                else:
                    ratio = stability_mean / remapping_mean  
                    print(f"    Stability > Remapping (Stability is {ratio:.1f}x larger)")
                    print(f"    → Place fields move MORE within same environment (unexpected!)")
        
        # Cross-stage statistical comparison
        print(f"\n{'='*80}")
        print("CROSS-STAGE STATISTICAL COMPARISON")
        print(f"{'='*80}")
        
        # Home cage stability across stages
        if (has_stability and 'anova_statistics' in self.stability_results):
            stability_stats = self.stability_results['anova_statistics'].get('home_cage_anova')
            if stability_stats is not None:
                print(f"\nHome Cage Stability Across Stages:")
                print(f"  Test: One-way ANOVA")
                print(f"  F({stability_stats['df_between']:.0f}, {stability_stats['df_within']:.0f}) = {stability_stats['f_stat']:.4f}")
                print(f"  p-value: {stability_stats['p_value']:.4f}")
                print(f"  Total n: {stability_stats['n_total']}")
                
                if stability_stats['p_value'] < 0.05:
                    print(f"  Result: Significant difference in stability across stages (p < 0.05)")
                else:
                    print(f"  Result: No significant difference in stability across stages (p ≥ 0.05)")
            else:
                print("\nHome Cage Stability: Insufficient data for cross-stage comparison")
        
        # Home to empty remapping across stages
        if (has_remapping and 'anova_statistics' in self.remapping_results):
            remapping_stats = self.remapping_results['anova_statistics'].get('home_to_empty_anova')
            if remapping_stats is not None:
                print(f"\nHome → Empty Remapping Across Stages:")
                print(f"  Test: One-way ANOVA")
                print(f"  F({remapping_stats['df_between']:.0f}, {remapping_stats['df_within']:.0f}) = {remapping_stats['f_stat']:.4f}")
                print(f"  p-value: {remapping_stats['p_value']:.4f}")
                print(f"  Total n: {remapping_stats['n_total']}")
                
                if remapping_stats['p_value'] < 0.05:
                    print(f"  Result: Significant difference in remapping across stages (p < 0.05)")
                else:
                    print(f"  Result: No significant difference in remapping across stages (p ≥ 0.05)")
            else:
                print("\nHome → Empty Remapping: Insufficient data for cross-stage comparison")
        
        # Overall interpretation
        print(f"\n{'='*80}")
        print("INTERPRETATION GUIDE")
        print(f"{'='*80}")
        print("• Lower stability values = Place fields are MORE stable (less movement)")
        print("• Higher remapping values = Place fields change MORE between environments")
        print("• Typically expect: Remapping > Stability (fields stable within environment,")
        print("  but change between different environments)")
        print("• Unexpected pattern: Stability > Remapping (may indicate issues with")
        print("  experimental setup or interesting biological phenomenon)")
        
    def save_stability_results(self, save_path=None, filename=None):
        """
        Save stability and remapping results to disk for longitudinal tracking
        
        Parameters:
        -----------
        save_path : str, optional
            Directory to save results. If None, uses self.data_paths
        filename : str, optional
            Custom filename. If None, auto-generates with timestamp
        
        Returns:
        --------
        full_path : str
            Path where results were saved
        """
        import pickle
        import os
        from datetime import datetime
        
        if not hasattr(self, 'stability_results'):
            print("No stability results to save. Run calculate_place_field_stability() first.")
            return None
        
        # Determine save path
        if save_path is None:
            save_path = self.data_paths if isinstance(self.data_paths, str) else self.data_paths[0]
        
        os.makedirs(save_path, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stage_name = self.stages[0] if len(self.stages) > 0 else 'unknown'
            filename = f"stability_results_{stage_name}_{timestamp}.pkl"
        
        full_path = os.path.join(save_path, filename)
        
        # Prepare data package
        save_dict = {
            'stability_results': self.stability_results,
            'remapping_results': self.remapping_results if hasattr(self, 'remapping_results') else None,
            'stages': self.stages,
            'environments': self.environments,
            'data_paths': self.data_paths,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'place_bin_size': getattr(self, 'place_bin_size', 2),
                'use_reliable_cells': True,  # Store your analysis parameters
                'convert_to_cm': True
            }
        }
        
        # Save to pickle
        with open(full_path, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n{'='*60}")
        print("STABILITY RESULTS SAVED")
        print(f"{'='*60}")
        print(f"Location: {full_path}")
        print(f"Timestamp: {save_dict['timestamp']}")
        print(f"Stages included: {self.stages}")
        
        # Print summary of what was saved
        if hasattr(self, 'stability_results'):
            stage_idx = 0
            if stage_idx in self.stability_results['empty_stability']:
                print(f"Empty cage: {len(self.stability_results['empty_stability'][stage_idx])} cells")
            if stage_idx in self.stability_results['home_stability']:
                print(f"Home cage: {len(self.stability_results['home_stability'][stage_idx])} cells")
        
        return full_path

    def load_stability_results(self, file_path):
        """
        Load previously saved stability results
        
        Parameters:
        -----------
        file_path : str
            Path to saved .pkl file
            
        Returns:
        --------
        loaded_data : dict
            Dictionary containing all saved results and metadata
        """
        import pickle
        import os
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Restore to class attributes
        self.stability_results = loaded_data['stability_results']
        self.remapping_results = loaded_data.get('remapping_results')
        self.stages = loaded_data['stages']
        self.environments = loaded_data['environments']
        
        print(f"\n{'='*60}")
        print("STABILITY RESULTS LOADED")
        print(f"{'='*60}")
        print(f"From: {file_path}")
        print(f"Original timestamp: {loaded_data['timestamp']}")
        print(f"Stages: {loaded_data['stages']}")
        print(f"Original data paths: {loaded_data['data_paths']}")
        
        return loaded_data

    def compare_stability_across_sessions(session_files, session_names=None, save_path=None):
        """
        Compare stability results across multiple recording sessions
        
        Parameters:
        -----------
        session_files : list of str
            Paths to saved stability results .pkl files (one per session)
        session_names : list of str, optional
            Custom names for each session. If None, uses filenames
        save_path : str, optional
            Where to save comparison plots
            
        Returns:
        --------
        comparison_results : dict
            Statistical comparison across sessions
        """
        import pickle
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        import pandas as pd
        
        if session_names is None:
            session_names = [os.path.basename(f).replace('.pkl', '') for f in session_files]
        
        print(f"\n{'='*60}")
        print(f"COMPARING STABILITY ACROSS {len(session_files)} SESSIONS")
        print(f"{'='*60}")
        
        # Load all session data
        all_sessions_data = []
        for file_path, session_name in zip(session_files, session_names):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                all_sessions_data.append({
                    'name': session_name,
                    'data': data,
                    'timestamp': data['timestamp']
                })
            print(f"Loaded: {session_name} ({data['timestamp']})")
        
        # Extract stability data for comparison
        empty_stability_by_session = []
        home_stability_by_session = []
        empty_labels = []
        home_labels = []
        
        for session in all_sessions_data:
            session_name = session['name']
            results = session['data']['stability_results']
            stage_idx = 0  # Assuming single stage per session
            
            # Empty cage stability
            if stage_idx in results['empty_stability'] and len(results['empty_stability'][stage_idx]) > 0:
                empty_data = results['empty_stability'][stage_idx]
                empty_stability_by_session.extend(empty_data)
                empty_labels.extend([session_name] * len(empty_data))
            
            # Home cage stability
            if stage_idx in results['home_stability'] and len(results['home_stability'][stage_idx]) > 0:
                home_data = results['home_stability'][stage_idx]
                home_stability_by_session.extend(home_data)
                home_labels.extend([session_name] * len(home_data))
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # EMPTY CAGE COMPARISON
        if len(empty_stability_by_session) > 0:
            df_empty = pd.DataFrame({
                'Stability (cm)': empty_stability_by_session,
                'Session': empty_labels
            })
            
            # Violin plot
            sns.violinplot(data=df_empty, x='Session', y='Stability (cm)', ax=axes[0, 0])
            axes[0, 0].set_title('Empty Cage Stability Across Sessions')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Box plot
            sns.boxplot(data=df_empty, x='Session', y='Stability (cm)', ax=axes[0, 1])
            axes[0, 1].set_title('Empty Cage Stability Across Sessions (Boxplot)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Statistical test
            if len(set(empty_labels)) >= 2:
                # One-way ANOVA
                groups = [df_empty[df_empty['Session'] == s]['Stability (cm)'].values 
                        for s in df_empty['Session'].unique()]
                f_stat, p_val = stats.f_oneway(*groups)
                
                axes[0, 0].text(0.5, 0.95, f'ANOVA: F={f_stat:.2f}, p={p_val:.4f}',
                            transform=axes[0, 0].transAxes, ha='center', va='top',
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
        
        # HOME CAGE COMPARISON
        if len(home_stability_by_session) > 0:
            df_home = pd.DataFrame({
                'Stability (cm)': home_stability_by_session,
                'Session': home_labels
            })
            
            # Violin plot
            sns.violinplot(data=df_home, x='Session', y='Stability (cm)', ax=axes[1, 0])
            axes[1, 0].set_title('Home Cage Stability Across Sessions')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Box plot
            sns.boxplot(data=df_home, x='Session', y='Stability (cm)', ax=axes[1, 1])
            axes[1, 1].set_title('Home Cage Stability Across Sessions (Boxplot)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Statistical test
            if len(set(home_labels)) >= 2:
                groups = [df_home[df_home['Session'] == s]['Stability (cm)'].values 
                        for s in df_home['Session'].unique()]
                f_stat, p_val = stats.f_oneway(*groups)
                
                axes[1, 0].text(0.5, 0.95, f'ANOVA: F={f_stat:.2f}, p={p_val:.4f}',
                            transform=axes[1, 0].transAxes, ha='center', va='top',
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig_path = os.path.join(save_path, 'stability_cross_session_comparison.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved to: {fig_path}")
        
        plt.show()
        
        # Print statistical summary
        print(f"\n{'='*60}")
        print("CROSS-SESSION STATISTICAL SUMMARY")
        print(f"{'='*60}")
        
        for session in all_sessions_data:
            session_name = session['name']
            results = session['data']['stability_results']
            stage_idx = 0
            
            print(f"\n{session_name}:")
            if stage_idx in results['empty_stability'] and len(results['empty_stability'][stage_idx]) > 0:
                empty_data = results['empty_stability'][stage_idx]
                print(f"  Empty: {np.mean(empty_data):.3f} ± {stats.sem(empty_data):.3f} cm (n={len(empty_data)})")
            
            if stage_idx in results['home_stability'] and len(results['home_stability'][stage_idx]) > 0:
                home_data = results['home_stability'][stage_idx]
                print(f"  Home:  {np.mean(home_data):.3f} ± {stats.sem(home_data):.3f} cm (n={len(home_data)})")
        
        return {
            'empty_df': df_empty if len(empty_stability_by_session) > 0 else None,
            'home_df': df_home if len(home_stability_by_session) > 0 else None,
            'session_data': all_sessions_data
        }