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
    
    def __init__(self, data_paths, stages, plot_flag=True):
        """
        Initialize PC_StabilityRemapping for home cage stability analysis
        
        Parameters:
        -----------
        data_paths : str or list
            Path(s) to data directories
        stages : list
            List of stage names (e.g., ['baseline1', 'baseline2', 'gestation1'])
        plot_flag : bool
            Whether to create plots (default: True)
        """
        self.data_paths = data_paths
        self.stages = stages
        self.plot_flag = plot_flag
        self.combined_preproc_data = None
        self.activity_maps_dict = None
        self.stability_results = None
        
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
            
            # Sort to ensure consistent ordering
            preproc_data_paths = sorted(preproc_data_paths)
            
            for path in preproc_data_paths:
                preproc_data = fm2p.read_h5(path)
                
                # Extract environment number from path
                # Try multiple patterns: 'env2', '\2\', '/2/', etc.
                import re
                import os
                
                # Pattern 1: Look for \2\ or /2/ in path (folder structure)
                match = re.search(r'[/\\](\d+)[/\\]', path)
                
                if not match:
                    # Pattern 2: Look for 'env2' in filename
                    match = re.search(r'env(\d+)', path.lower())
                
                if match:
                    env_num = int(match.group(1))
                    # Map env2 (Home2) -> index 1, env4 (Home4) -> index 3
                    if env_num == 2:
                        env_key = 1
                    elif env_num == 4:
                        env_key = 3
                    elif env_num == 1:
                        env_key = 0  # Empty1
                    elif env_num == 3:
                        env_key = 2  # Empty3
                    else:
                        env_key = env_num - 1  # Default 0-indexing
                    
                    combined_data[0, env_key] = preproc_data
                    print(f"Loaded environment {env_num} -> index {env_key} from: {os.path.basename(path)}")
                else:
                    # Fallback: use order if no env number found
                    env_key = len(combined_data)
                    combined_data[0, env_key] = preproc_data
                    print(f"WARNING: Could not detect environment number from path: {path}")
                    print(f"         Using fallback index {env_key}")
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
        Calculate home cage place field stability (Home2 → Home4) using Earth Mover's Distance
        
        Parameters:
        -----------
        use_reliable_cells : bool, default True
            Whether to use only reliable place cells (cohens_d criterion)
        convert_to_cm : bool, default True
            Whether to convert distances from bins to cm
        place_bin_size : float, default 2
            Size of each spatial bin in cm
            
        Returns:
        --------
        stability_results : dict
            Contains home stability EMD values and cell IDs for all stages
        """
                
        if self.combined_preproc_data is None:
            raise ValueError("Must load preprocessed data first using load_preprocessed_data()")
        
        if self.activity_maps_dict is None:
            print("Activity maps not loaded. Extracting from preprocessed data...")
            self.extract_activity_maps()
        
        # Initialize results storage
        results = {
            'home_stability': {},   # {stage_idx: [EMD values]}
            'home_cell_ids': {}     # {stage_idx: [cell_indices]}
        }
        
        print("=== Home Cage Stability Analysis (Home2 → Home4) ===")
        print("Using Earth Mover's Distance (EMD) via POT library\n")
        
        # Get all available stage indices
        available_stages = set([key[0] for key in self.combined_preproc_data.keys()])
        print(f"Found {len(available_stages)} stages: {sorted(available_stages)}")
        
        # Loop through all stages
        for stage_idx in sorted(available_stages):
            print(f"\n{'='*60}")
            print(f"PROCESSING STAGE {stage_idx} ({self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'})")
            print(f"{'='*60}")
            
            # Check if Home2 (env 1) and Home4 (env 3) exist for this stage
            if (stage_idx, 1) not in self.combined_preproc_data or (stage_idx, 3) not in self.combined_preproc_data:
                print(f"Skipping stage {stage_idx}: Missing Home2 or Home4 data")
                continue
            
            if (stage_idx, 1) not in self.activity_maps_dict or (stage_idx, 3) not in self.activity_maps_dict:
                print(f"Skipping stage {stage_idx}: Missing activity maps for Home2 or Home4")
                continue
            
            # Initialize stage-specific results
            results['home_stability'][stage_idx] = []
            results['home_cell_ids'][stage_idx] = []
            
            # HOME CAGE STABILITY (Home2 vs Home4)
            print(f"\nCalculating home cage stability (Home2 → Home4)...")
            
            # Get place cell indices for home environments  
            home2_data = self.combined_preproc_data[stage_idx, 1]  # Home2
            home4_data = self.combined_preproc_data[stage_idx, 3]  # Home4
            
            if use_reliable_cells:
                home2_reliable = home2_data['place_cell_reliability']
                home4_reliable = home4_data['place_cell_reliability']
                home_valid_cells = home2_reliable & home4_reliable
                print(f"  Using reliable cells (cohens_d criterion)")
            else:
                home2_place_cells = home2_data['place_cell_inds']
                home4_place_cells = home4_data['place_cell_inds']
                home_valid_cells = home2_place_cells & home4_place_cells
                print(f"  Using multi-criteria place cells (sigSI & sigRel & hasPlaceField)")
            
            home_cell_indices = np.where(home_valid_cells)[0]
            print(f"  Found {len(home_cell_indices)} valid cells in both Home2 and Home4")
            
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
            
            print(f"  Calculated stability for {len(home_emd)} cells")
            if len(home_emd) > 0:
                unit_str = 'cm' if convert_to_cm else 'bins'
                print(f"  Mean ± SEM: {np.mean(home_emd):.3f} ± {stats.sem(home_emd):.3f} {unit_str}")
                print(f"  Median: {np.median(home_emd):.3f} {unit_str}")
                print(f"  Range: [{np.min(home_emd):.3f}, {np.max(home_emd):.3f}] {unit_str}")
        
        # Store results in class
        self.stability_results = results
        
        print(f"\n{'='*60}")
        print(f"STABILITY ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Processed {len(results['home_stability'])} stages")
        
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
    
    def plot_home_stability_violin(self, convert_to_cm=True, save_path=None):
        """
        Plot home cage stability (Home2 → Home4) as violin plot for single session
        
        Parameters:
        -----------
        convert_to_cm : bool, default True
            Whether to use cm units
        save_path : str, optional
            Directory path where the figure will be saved as 'home_stability_violin.png'
            If None, figure is only displayed
        """

        if not hasattr(self, 'stability_results') or self.stability_results is None:
            print("No stability results found. Run calculate_place_field_stability() first.")
            return
        
        results = self.stability_results
        unit_str = 'cm' if convert_to_cm else 'bins'
        
        # Get data for all stages
        all_home_stability = []
        all_stage_labels = []
        
        for stage_idx in sorted(results['home_stability'].keys()):
            home_stability = results['home_stability'][stage_idx]
            if len(home_stability) > 0:
                stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                all_home_stability.extend(home_stability)
                all_stage_labels.extend([stage_name] * len(home_stability))
        
        if len(all_home_stability) == 0:
            print("No home stability data available for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Stability Distance': all_home_stability,
            'Stage': all_stage_labels
        })
        
        # Violin plot
        sns.violinplot(data=df, x='Stage', y='Stability Distance', ax=ax)
        ax.set_ylabel(f'Place Field Stability\nEMD Distance ({unit_str})')
        ax.set_title('Home Cage Place Field Stability (Home2 → Home4)')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # Add sample sizes
        unique_stages = df['Stage'].unique()
        for i, stage in enumerate(unique_stages):
            stage_data = df[df['Stage'] == stage]['Stability Distance']
            n = len(stage_data)
            mean_val = np.mean(stage_data)
            
            # Add n= above violin
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(i, y_pos, f'n={n}', ha='center', va='top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure if directory path provided
        if save_path is not None:
            import os
            filename = "home_stability_violin.png"
            full_path = os.path.join(save_path, filename)
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("HOME CAGE STABILITY SUMMARY")
        print("="*60)
        
        for stage_idx in sorted(results['home_stability'].keys()):
            home_stability = results['home_stability'][stage_idx]
            if len(home_stability) > 0:
                stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                print(f"\n{stage_name}:")
                print(f"  • {len(home_stability)} place cells analyzed")
                print(f"  • Mean ± SEM: {np.mean(home_stability):.3f} ± {stats.sem(home_stability):.3f} {unit_str}")
                print(f"  • Median: {np.median(home_stability):.3f} {unit_str}")
                print(f"  • Range: [{np.min(home_stability):.3f}, {np.max(home_stability):.3f}] {unit_str}")
        
        print(f"\nInterpretation: Lower EMD values indicate more stable place fields")
    
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
    
    def plot_home_stability_histograms(self, convert_to_cm=True):
        """
        Plot EMD histograms for home cage stability across stages
        Each stage shown in different color with statistics
        
        Parameters:
        -----------
        convert_to_cm : bool, default True
            Whether to use cm units in labels
        """
        
        if not hasattr(self, 'stability_results') or self.stability_results is None:
            print("No stability results found. Run calculate_place_field_stability() first.")
            return
        
        results = self.stability_results
        unit_str = 'EMD (cm)' if convert_to_cm else 'EMD (bins)'
        
        # Get all available stages with data
        available_stages = [s for s in results['home_stability'].keys() 
                        if len(results['home_stability'][s]) > 0]
        
        if not available_stages:
            print("No home stability data available")
            return
        
        print(f"\n{'='*60}")
        print(f"HOME CAGE STABILITY HISTOGRAMS BY STAGE")
        print(f"{'='*60}")
        
        # Define colors for different stages
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot histogram for each stage
        for i, stage_idx in enumerate(available_stages):
            home_data = results['home_stability'][stage_idx]
            stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
            color = colors[i % len(colors)]
            
            # Plot histogram
            ax.hist(home_data, bins=20, alpha=0.6, color=color, 
                label=f'{stage_name} (n={len(home_data)})', density=True)
            
            # Print stats to console
            print(f"\n{stage_name}:")
            print(f"  n = {len(home_data)}")
            print(f"  Mean ± SEM: {np.mean(home_data):.3f} ± {stats.sem(home_data):.3f}")
            print(f"  Median: {np.median(home_data):.3f}")
            print(f"  Range: [{np.min(home_data):.3f}, {np.max(home_data):.3f}]")
            print(f"  Color: {color}")
        
        ax.legend()
        ax.set_xlabel(unit_str)
        ax.set_ylabel('Density')
        ax.set_title(f'Home Cage Stability Distribution (Home2 → Home4)\nAcross Recording Stages')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print overall summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Available stages: {[self.stages[s] if s < len(self.stages) else f'Stage_{s}' for s in available_stages]}")
        print(f"Colors used: {[colors[i % len(colors)] for i in range(len(available_stages))]}")
        
        # Combined statistics
        all_data = np.concatenate([results['home_stability'][s] for s in available_stages])
        print(f"\nCombined across all stages (n={len(all_data)}):")
        print(f"  Mean ± SEM: {np.mean(all_data):.3f} ± {stats.sem(all_data):.3f}")
        print(f"  Median: {np.median(all_data):.3f}")
        
    def save_stability_results(self, save_path=None, filename=None):
        """
        Save home cage stability results to disk for longitudinal tracking  # CHANGED: Updated docstring
        
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
        
        if not hasattr(self, 'stability_results') or self.stability_results is None:  # CHANGED: Added 'or self.stability_results is None'
            print("No stability results to save. Run calculate_place_field_stability() first.")
            return None
        
        # Determine save path
        if save_path is None:
            save_path = self.data_paths if isinstance(self.data_paths, str) else self.data_paths[0]
        
        os.makedirs(save_path, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # CHANGED: Added seconds
            stage_name = self.stages[0] if len(self.stages) > 0 else 'unknown'
            filename = f"home_stability_{stage_name}_{timestamp}.pkl"  # CHANGED: Renamed and changed to .pkl
        
        full_path = os.path.join(save_path, filename)
        
        # Prepare data package
        save_dict = {
            'home_stability': self.stability_results['home_stability'],  # CHANGED: Only home stability
            'home_cell_ids': self.stability_results['home_cell_ids'],    # CHANGED: Only home cell IDs
            'stages': self.stages,                                        # KEEP
            'data_paths': self.data_paths,                                # KEEP
            'timestamp': datetime.now().isoformat(),                      # KEEP
            'metadata': {
                'analysis_type': 'home_cage_stability',                   # CHANGED: Added analysis type
                'comparison': 'Home2_to_Home4',                           # CHANGED: Added comparison info
                'place_bin_size': 2,                                      # CHANGED: Simplified
                'use_reliable_cells': True,                               # KEEP
                'convert_to_cm': True                                     # KEEP
            }
        }
        
        # CHANGED: Save to pickle instead of h5
        with open(full_path, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n{'='*60}")
        print("HOME STABILITY RESULTS SAVED")  # CHANGED: Updated message
        print(f"{'='*60}")
        print(f"Location: {full_path}")
        print(f"Timestamp: {save_dict['timestamp']}")
        print(f"Stages included: {self.stages}")
        
        # CHANGED: Simplified summary printing
        for stage_idx in self.stability_results['home_stability'].keys():
            n_cells = len(self.stability_results['home_stability'][stage_idx])
            stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
            print(f"  {stage_name}: {n_cells} cells")
        
        return full_path
    
    def load_stability_results(self, file_path):
        """
        Load previously saved home cage stability results
        
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
        self.stability_results = {
            'home_stability': loaded_data['home_stability'],
            'home_cell_ids': loaded_data['home_cell_ids']
        }
        self.stages = loaded_data['stages']
        
        print(f"\n{'='*60}")
        print("HOME STABILITY RESULTS LOADED")
        print(f"{'='*60}")
        print(f"From: {file_path}")
        print(f"Original timestamp: {loaded_data['timestamp']}")
        print(f"Stages: {loaded_data['stages']}")
        print(f"Original data paths: {loaded_data['data_paths']}")
        
        # Print loaded data summary
        for stage_idx in loaded_data['home_stability'].keys():
            n_cells = len(loaded_data['home_stability'][stage_idx])
            stage_name = loaded_data['stages'][stage_idx] if stage_idx < len(loaded_data['stages']) else f'Stage_{stage_idx}'
            mean_emd = np.mean(loaded_data['home_stability'][stage_idx])
            print(f"  {stage_name}: {n_cells} cells, mean EMD = {mean_emd:.3f} cm")
        
        return loaded_data
    
    @staticmethod
    def load_multiple_sessions(session_files, session_names=None):
        """
        Load home stability results from multiple .pkl files
        
        Parameters:
        -----------
        session_files : list of str
            Paths to .pkl files for each session
        session_names : list of str, optional
            Names for each session (e.g., ["Baseline1", "Baseline2", "Gestation#1"])
            If None, uses filenames without extension
            
        Returns:
        --------
        sessions_data : dict
            Dictionary with structure:
            {
                session_name: {
                    'home_stability': array of EMD values,
                    'home_cell_ids': array of cell indices,
                    'n_cells': int,
                    'mean_emd': float,
                    'sem_emd': float,
                    'timestamp': str,
                    'data_paths': str
                }
            }
        """
        import pickle
        import os
        
        if session_names is None:
            session_names = [os.path.splitext(os.path.basename(f))[0] for f in session_files]
        
        if len(session_files) != len(session_names):
            raise ValueError(f"Number of session files ({len(session_files)}) must match number of session names ({len(session_names)})")
        
        sessions_data = {}
        
        print(f"\n{'='*60}")
        print(f"LOADING {len(session_files)} SESSIONS")
        print(f"{'='*60}")
        
        for file_path, session_name in zip(session_files, session_names):
            if not os.path.exists(file_path):
                print(f"WARNING: File not found: {file_path}")
                continue
            
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Extract home stability data (assuming single stage per session, stage_idx=0)
            stage_idx = 0
            
            if stage_idx not in loaded_data['home_stability']:
                print(f"WARNING: No data for stage {stage_idx} in {session_name}")
                continue
            
            home_stability = loaded_data['home_stability'][stage_idx]
            home_cell_ids = loaded_data['home_cell_ids'][stage_idx]
            
            # Store organized session data
            sessions_data[session_name] = {
                'home_stability': home_stability,
                'home_cell_ids': home_cell_ids,
                'n_cells': len(home_stability),
                'mean_emd': np.mean(home_stability) if len(home_stability) > 0 else np.nan,
                'sem_emd': stats.sem(home_stability) if len(home_stability) > 0 else np.nan,
                'median_emd': np.median(home_stability) if len(home_stability) > 0 else np.nan,
                'timestamp': loaded_data['timestamp'],
                'data_paths': loaded_data['data_paths']
            }
            
            print(f"\n{session_name}:")
            print(f"  File: {os.path.basename(file_path)}")
            print(f"  Timestamp: {loaded_data['timestamp']}")
            print(f"  Cells: {len(home_stability)}")
            print(f"  Mean ± SEM: {sessions_data[session_name]['mean_emd']:.3f} ± {sessions_data[session_name]['sem_emd']:.3f} cm")
        
        print(f"\n{'='*60}")
        print(f"Successfully loaded {len(sessions_data)} sessions")
        print(f"{'='*60}")
        
        return sessions_data
    
    @staticmethod
    def plot_home_stability_by_session(session_files, session_names=None, save_path=None, convert_to_cm=True):
        """
        Compare home cage stability across multiple sessions with simple line plot
        
        Parameters:
        -----------
        session_files : list of str
            Paths to saved .pkl files for each session
        session_names : list of str, optional
            Custom names for sessions (e.g., ["Baseline1", "Baseline2", "Gestation#1"])
        save_path : str, optional
            Directory to save the comparison plot
        convert_to_cm : bool, default True
            Whether to use cm units
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        import pandas as pd
        import os
        
        # Load all sessions
        sessions_data = PC_StabilityRemapping.load_multiple_sessions(session_files, session_names)
        
        if len(sessions_data) == 0:
            print("No sessions loaded. Cannot create comparison plot.")
            return
        
        unit_str = 'cm' if convert_to_cm else 'bins'
        
        # Prepare data for plotting
        session_order = list(sessions_data.keys())
        means = [sessions_data[s]['mean_emd'] for s in session_order]
        sems = [sessions_data[s]['sem_emd'] for s in session_order]
        n_cells = [sessions_data[s]['n_cells'] for s in session_order]
        
        # Create simple figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Simple line plot with error bars
        x_pos = np.arange(len(session_order))
        
        ax.errorbar(x_pos, means, yerr=sems, 
                    fmt='o-', markersize=8, linewidth=2, 
                    color='steelblue', ecolor='steelblue',
                    capsize=6, capthick=2)
        
        # Labels and styling
        ax.set_ylabel(f'Mean Home Cage Stability \n EMD Distance ({unit_str})', fontsize=12, fontweight = "bold")
        ax.set_ylim([0, 10])
        ax.set_xlabel('Session', fontsize=12, fontweight = "bold")
        ax.set_title('Home Cage Place Field Stability Across Sessions', fontsize=16, fontweight = "bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(session_order, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add sample sizes above points
        for i, (x, mean_val, sem_val, n) in enumerate(zip(x_pos, means, sems, n_cells)):
            ax.text(x, mean_val + sem_val + 0.1, f'n={n}', 
                    ha='center', va='bottom', fontsize=9)
        
        # # Perform statistical test
        # if len(sessions_data) >= 2:
        #     # Prepare data
        #     all_emd_values = []
        #     all_session_labels = []
            
        #     for session_name in session_order:
        #         emd_values = sessions_data[session_name]['home_stability']
        #         all_emd_values.extend(emd_values)
        #         all_session_labels.extend([session_name] * len(emd_values))
            
        #     df = pd.DataFrame({
        #         'EMD': all_emd_values,
        #         'Session': all_session_labels
        #     })
            
        #     # ANOVA for overall difference
        #     from statsmodels.formula.api import ols
        #     from statsmodels.stats.anova import anova_lm
            
        #     model = ols('EMD ~ C(Session)', data=df).fit()
        #     anova_result = anova_lm(model, typ=2)
            
        #     f_stat = anova_result['F'].iloc[0]
        #     p_value = anova_result['PR(>F)'].iloc[0]
            
        #     # Add result to plot
        #     sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        #     ax.text(0.98, 0.98, f'p = {p_value:.4f} {sig_marker}', 
        #             transform=ax.transAxes, ha='right', va='top',
        #             fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        #     # Print results
        #     print(f"\n{'='*60}")
        #     print("ANOVA RESULTS")
        #     print(f"{'='*60}")
        #     print(f"F-statistic: {f_stat:.4f}")
        #     print(f"p-value: {p_value:.4f}")
        #     print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        plt.tight_layout()
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filename = "home_stability_comparison.png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {full_path}")
        
        plt.show()
    
    # @staticmethod
    # def plot_home_stability_by_session(session_files, session_names=None, save_path=None, convert_to_cm=True):
    #     """
    #     Compare home cage stability across multiple sessions with line plot
        
    #     Parameters:
    #     -----------
    #     session_files : list of str
    #         Paths to saved .pkl files for each session
    #     session_names : list of str, optional
    #         Custom names for sessions (e.g., ["Baseline1", "Baseline2", "Gestation#1"])
    #         If None, uses filenames
    #     save_path : str, optional
    #         Directory to save the comparison plot
    #     convert_to_cm : bool, default True
    #         Whether to use cm units
            
    #     Creates:
    #     --------
    #     - Line plot with points showing mean ± SEM for each session
    #     - ANOVA test comparing all sessions
    #     - Sample sizes displayed
    #     - Statistical significance annotations
    #     """
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     from scipy import stats
    #     import pandas as pd
    #     import os
        
    #     # Load all sessions
    #     sessions_data = PC_StabilityRemapping.load_multiple_sessions(session_files, session_names)
        
    #     if len(sessions_data) == 0:
    #         print("No sessions loaded. Cannot create comparison plot.")
    #         return
        
    #     unit_str = 'cm' if convert_to_cm else 'bins'
        
    #     # Prepare data for plotting
    #     session_order = list(sessions_data.keys())
    #     means = [sessions_data[s]['mean_emd'] for s in session_order]
    #     sems = [sessions_data[s]['sem_emd'] for s in session_order]
    #     n_cells = [sessions_data[s]['n_cells'] for s in session_order]
        
    #     # Create figure
    #     fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
    #     # Line plot with error bars
    #     x_pos = np.arange(len(session_order))
        
    #     # Plot line connecting points
    #     ax.plot(x_pos, means, marker='o', markersize=10, linewidth=2.5, 
    #             color='steelblue', label='Mean ± SEM', zorder=3)
        
    #     # Add error bars (SEM)
    #     ax.errorbar(x_pos, means, yerr=sems, fmt='none', ecolor='steelblue', 
    #                 elinewidth=2, capsize=8, capthick=2, alpha=0.7, zorder=2)
        
    #     # Add individual points as scatter for emphasis
    #     ax.scatter(x_pos, means, s=150, color='steelblue', edgecolors='navy', 
    #             linewidth=2, zorder=4, alpha=0.8)
        
    #     # Styling
    #     ax.set_ylabel(f'Mean Home Cage Stability\nEMD Distance ({unit_str})', fontsize=13, fontweight='bold')
    #     ax.set_xlabel('Recording Session', fontsize=13, fontweight='bold')
    #     ax.set_title('Home Cage Place Field Stability Across Sessions', 
    #                 fontsize=15, fontweight='bold', pad=20)
    #     ax.set_xticks(x_pos)
    #     ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=11)
    #     ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    #     ax.set_xlim(-0.5, len(session_order) - 0.5)
        
    #     # Add sample sizes above each point
    #     for i, (x, mean_val, sem_val, n) in enumerate(zip(x_pos, means, sems, n_cells)):
    #         # Sample size above the error bar
    #         y_offset = mean_val + sem_val + (max(means) - min(means)) * 0.05
    #         ax.text(x, y_offset, f'n={n}', ha='center', va='bottom', 
    #                 fontweight='bold', fontsize=10, 
    #                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
    #                         edgecolor='steelblue', alpha=0.8))
            
    #         # Mean value next to the point
    #         ax.text(x + 0.15, mean_val, f'{mean_val:.2f}', ha='left', va='center', 
    #                 fontsize=9, fontweight='bold', color='navy')
        
    #     # Perform ANOVA if we have 2+ sessions
    #     if len(sessions_data) >= 2:
    #         # Prepare data for ANOVA
    #         all_emd_values = []
    #         all_session_labels = []
            
    #         for session_name in session_order:
    #             emd_values = sessions_data[session_name]['home_stability']
    #             all_emd_values.extend(emd_values)
    #             all_session_labels.extend([session_name] * len(emd_values))
            
    #         # Create DataFrame for ANOVA
    #         df = pd.DataFrame({
    #             'EMD': all_emd_values,
    #             'Session': all_session_labels
    #         })
            
    #         # One-way ANOVA
    #         from statsmodels.formula.api import ols
    #         from statsmodels.stats.anova import anova_lm
            
    #         model = ols('EMD ~ C(Session)', data=df).fit()
    #         anova_result = anova_lm(model, typ=2)
            
    #         f_stat = anova_result['F'].iloc[0]
    #         p_value = anova_result['PR(>F)'].iloc[0]
    #         df_between = anova_result['df'].iloc[0]
    #         df_within = anova_result['df'].iloc[1]
            
    #         # Add ANOVA results to plot
    #         stats_text = f'One-way ANOVA: F({df_between:.0f},{df_within:.0f}) = {f_stat:.2f}, p = {p_value:.4f}'
    #         if p_value < 0.001:
    #             stats_text += ' ***'
    #             sig_color = 'red'
    #         elif p_value < 0.01:
    #             stats_text += ' **'
    #             sig_color = 'orange'
    #         elif p_value < 0.05:
    #             stats_text += ' *'
    #             sig_color = 'gold'
    #         else:
    #             sig_color = 'lightgray'
            
    #         ax.text(0.5, 0.98, stats_text, 
    #                 transform=ax.transAxes, ha='center', va='top',
    #                 fontsize=12, fontweight='bold',
    #                 bbox=dict(boxstyle="round,pad=0.6", facecolor=sig_color, 
    #                         edgecolor='black', linewidth=2, alpha=0.9))
            
    #         # Print ANOVA results to console
    #         print(f"\n{'='*60}")
    #         print("STATISTICAL COMPARISON (ONE-WAY ANOVA)")
    #         print(f"{'='*60}")
    #         print(f"F({df_between:.0f}, {df_within:.0f}) = {f_stat:.4f}")
    #         print(f"p-value = {p_value:.4f}")
            
    #         if p_value < 0.05:
    #             print("Result: SIGNIFICANT difference between sessions (p < 0.05)")
                
    #             # Post-hoc pairwise comparisons if significant
    #             from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
    #             print("\nPost-hoc pairwise comparisons (Tukey HSD):")
    #             tukey = pairwise_tukeyhsd(endog=all_emd_values, groups=all_session_labels, alpha=0.05)
    #             print(tukey)
    #         else:
    #             print("Result: No significant difference between sessions (p ≥ 0.05)")
        
    #     plt.tight_layout()
        
    #     # Save figure if path provided
    #     if save_path is not None:
    #         os.makedirs(save_path, exist_ok=True)
    #         filename = "home_stability_session_comparison.png"
    #         full_path = os.path.join(save_path, filename)
    #         plt.savefig(full_path, dpi=300, bbox_inches='tight')
    #         print(f"\nFigure saved to: {full_path}")
        
    #     plt.show()
        
    #     # Print session summary table
    #     print(f"\n{'='*60}")
    #     print("SESSION SUMMARY TABLE")
    #     print(f"{'='*60}")
    #     print(f"{'Session':<20} {'n cells':<10} {'Mean ± SEM (cm)':<20} {'Median (cm)':<15}")
    #     print("-" * 60)
    #     for session_name in session_order:
    #         data = sessions_data[session_name]
    #         print(f"{session_name:<20} {data['n_cells']:<10} {data['mean_emd']:.3f} ± {data['sem_emd']:.3f}{'':<8} {data['median_emd']:.3f}")
            
    def print_home_stability_summary(self, convert_to_cm=True):
        """
        Print clean summary of home cage stability results for current instance
        
        Parameters:
        -----------
        convert_to_cm : bool, default True
            Whether to display values in cm units
        """
        
        if not hasattr(self, 'stability_results') or self.stability_results is None:
            print("No stability results found. Run calculate_place_field_stability() first.")
            return
        
        results = self.stability_results
        unit_str = 'cm' if convert_to_cm else 'bins'
        
        print("\n" + "="*70)
        print("HOME CAGE STABILITY SUMMARY (Home2 → Home4)")
        print("="*70)
        print("Lower EMD values = More stable place fields")
        print("="*70)
        
        # Get all stages with data
        stages_with_data = [s for s in results['home_stability'].keys() 
                        if len(results['home_stability'][s]) > 0]
        
        if not stages_with_data:
            print("No stability data available.")
            return
        
        # Print summary for each stage
        for stage_idx in sorted(stages_with_data):
            stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
            home_stability = results['home_stability'][stage_idx]
            
            print(f"\n{stage_name.upper()}:")
            print("-" * 70)
            print(f"  Number of cells analyzed: {len(home_stability)}")
            print(f"  Mean ± SEM:  {np.mean(home_stability):.3f} ± {stats.sem(home_stability):.3f} {unit_str}")
            print(f"  Median:      {np.median(home_stability):.3f} {unit_str}")
            print(f"  Range:       [{np.min(home_stability):.3f}, {np.max(home_stability):.3f}] {unit_str}")
            print(f"  Std Dev:     {np.std(home_stability):.3f} {unit_str}")
            
            # Quartiles
            q25, q75 = np.percentile(home_stability, [25, 75])
            print(f"  25th-75th percentile: [{q25:.3f}, {q75:.3f}] {unit_str}")
        
        # If multiple stages, print comparison
        if len(stages_with_data) > 1:
            print(f"\n{'='*70}")
            print("CROSS-STAGE COMPARISON")
            print(f"{'='*70}")
            
            # Prepare data for ANOVA
            all_emd_values = []
            all_stage_labels = []
            
            for stage_idx in stages_with_data:
                stage_name = self.stages[stage_idx] if stage_idx < len(self.stages) else f'Stage_{stage_idx}'
                emd_values = results['home_stability'][stage_idx]
                all_emd_values.extend(emd_values)
                all_stage_labels.extend([stage_name] * len(emd_values))
            
            # One-way ANOVA
            from statsmodels.formula.api import ols
            from statsmodels.stats.anova import anova_lm
            
            df = pd.DataFrame({
                'EMD': all_emd_values,
                'Stage': all_stage_labels
            })
            
            model = ols('EMD ~ C(Stage)', data=df).fit()
            anova_result = anova_lm(model, typ=2)
            
            f_stat = anova_result['F'].iloc[0]
            p_value = anova_result['PR(>F)'].iloc[0]
            df_between = anova_result['df'].iloc[0]
            df_within = anova_result['df'].iloc[1]
            
            print(f"\nOne-way ANOVA:")
            print(f"  F({df_between:.0f}, {df_within:.0f}) = {f_stat:.4f}")
            print(f"  p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  Result: SIGNIFICANT difference between stages (p < 0.05)")
            else:
                print(f"  Result: No significant difference between stages (p ≥ 0.05)")
        
        print(f"\n{'='*70}")
        

def compare_home_stability_across_sessions(session_files, session_names=None, save_path=None):
    """
    Standalone function to compare home cage stability across multiple recording sessions
    
    This is a convenience wrapper around PC_StabilityRemapping.plot_home_stability_by_session()
    
    Parameters:
    -----------
    session_files : list of str
        Paths to saved .pkl files for each session
        Example: [
            r"path/to/baseline1_results.pkl",
            r"path/to/baseline2_results.pkl", 
            r"path/to/gestation1_results.pkl"
        ]
    session_names : list of str, optional
        Custom names for sessions (e.g., ["Baseline1", "Baseline2", "Gestation#1"])
        If None, uses filenames
    save_path : str, optional
        Directory to save comparison plots and results
        
    Returns:
    --------
    sessions_data : dict
        Dictionary containing loaded data and statistics for all sessions
    """
    
    print("\n" + "="*70)
    print("CROSS-SESSION HOME CAGE STABILITY COMPARISON")
    print("="*70)
    
    # Create the comparison plot
    PC_StabilityRemapping.plot_home_stability_by_session(
        session_files=session_files,
        session_names=session_names,
        save_path=save_path,
        convert_to_cm=True
    )
    
    # Load and return the data for further analysis if needed
    sessions_data = PC_StabilityRemapping.load_multiple_sessions(
        session_files=session_files,
        session_names=session_names
    )
    
    return sessions_data