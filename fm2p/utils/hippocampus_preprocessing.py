import os
import numpy as np
import fm2p
from fm2p.utils.place_cells import plot_place_cell_maps


def hippocampal_preprocess(cfg_path):

    cfg = fm2p.read_yaml(cfg_path)
    # recording_names = cfg['hp_empty_recs']
    recording_names = cfg['hp_home_recs']
    # recording_names = cfg['hp_empty_recs'] + cfg['hp_home_recs']
    num_recordings = len(recording_names)
    
    cohens_thresh = cfg['cohens_d']
    
    for rnum, rname in enumerate(recording_names):
        
        full_rname = rnum+1
        # r_name = rname.split('\\')[-1]
        # rpath = cfg['spath']
        rpath = os.path.join(cfg['spath'], "Processed",str(rname))

        # Topdown camera files
        possible_topdown_videos = fm2p.find('*.mp4', rpath, MR=False)
        topdown_video = fm2p.filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)
    
        # # Run pose estimation
        # fm2p.run_pose_estimation(
        #     topdown_video,
        #     project_cfg=cfg['top_DLC_project'],
        #     filter=False
        # )
    
        topdown_pts_path = fm2p.find('*DLC*.h5', rpath, MR=True)
        # topdown_pts_path = fm2p.find('*topDLC_resnet50*.h5', rpath, MR=True)
    
        F_path = fm2p.find('F.npy', rpath, MR=True)
        Fneu_path = fm2p.find('Fneu.npy', rpath, MR=True)
        suite2p_spikes = fm2p.find('spks.npy', rpath, MR=True)
        iscell_path = fm2p.find('iscell.npy', rpath, MR=True)
        stat_path = fm2p.find('stat.npy', rpath, MR=True)
        ops_path = fm2p.find('ops.npy', rpath, MR=True)
        
        
        # Topdown behavior and obstacle/arena tracking
        top_cam = fm2p.Topcam(rpath, '', cfg=cfg)
        top_cam.add_files(
            top_dlc_h5=topdown_pts_path,
            top_avi=topdown_video
        )
        arena_dict = top_cam.track_arena(no_pillar=True, recording_name=rname)
        pxls2cm = arena_dict['pxls2cm']
        top_xyl, top_tracking_dict = top_cam.track_body(pxls2cm)
        F = np.load(F_path, allow_pickle=True)
        Fneu = np.load(Fneu_path, allow_pickle=True)
        spks = np.load(suite2p_spikes, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)
        stat = np.load(stat_path, allow_pickle=True)
        ops =  np.load(ops_path, allow_pickle=True)
    
        twop_recording = fm2p.TwoP(rpath, '', cfg=cfg)
        twop_recording.add_data(
            F=F,
            Fneu=Fneu,
            spikes=spks,
            iscell=iscell
        )
        twop_dict = twop_recording.calc_dFF(neu_correction=0.7, oasis=False)
        dFF_transients = twop_recording.calc_dFF_transients()
        normspikes = twop_recording.normalize_spikes()
        recording_props = twop_recording.get_recording_props(
            stat=stat,
            ops=ops
        )

        twop_dt = 1./cfg['twop_rate']
        twopT = np.arange(0, np.size(twop_dict['s2p_spks'], 1)*twop_dt, twop_dt)
        twop_dict['twopT'] = twopT
        twop_dict['matlab_cellinds'] = np.arange(np.size(twop_dict['raw_F'],0))
        twop_dict['normspikes'] = normspikes
        twop_dict['dFF_transients'] = dFF_transients
    
        # All values in units of pixels or degrees (not cm or rads)
        # learx = top_tracking_dict['lear_x']
        # leary = top_tracking_dict['lear_y']
        # rearx = top_tracking_dict['rear_x']
        # reary = top_tracking_dict['rear_y']
        # yaw = top_tracking_dict['head_yaw_deg']
        
        pos_x = top_tracking_dict['smooth_x']
        pos_y = top_tracking_dict['smooth_y']

        # For rare instances when scanimage acquired more 2P frames than topdown camera.
        # This is usually by ~1 frame, but this will handle larger discrepancies.
        _len_diff = np.size(top_tracking_dict['smooth_x']) - np.size(twop_dict['s2p_spks'], 1)
        while _len_diff != 0:
            if _len_diff > 0:
                # top tracking is too long for spike data
                # learx = learx[:-1]
                # leary = leary[:-1]
                # rearx = rearx[:-1]
                # reary = reary[:-1]
                # yaw = yaw[:-1]

                top_tracking_dict['smooth_x'] = top_tracking_dict['smooth_x'][:-1]
                top_tracking_dict['smooth_y'] = top_tracking_dict['smooth_y'][:-1]
                top_tracking_dict['speed'] = top_tracking_dict['speed'][:-1]
            elif _len_diff < 0:

                # spike data is too long for top tracking
                twop_dict['twopT'] = twop_dict['twopT'][:-1]
                twop_dict['raw_F0'] = twop_dict['raw_F0'][:-1]
                twop_dict['raw_F'] = twop_dict['raw_F'][:-1]
                twop_dict['norm_F'] = twop_dict['norm_F'][:-1]
                twop_dict['raw_Fneu'] = twop_dict['raw_Fneu'][:-1]
                twop_dict['raw_dFF'] = twop_dict['raw_dFF'][:-1]
                twop_dict['norm_dFF'] = twop_dict['norm_dFF'][:-1]
                # twop_dict['denoised_dFF'] = twop_dict['denoised_dFF'][:-1]
                twop_dict['s2p_spks'] = twop_dict['s2p_spks'][:-1]
            _len_diff = np.size(top_tracking_dict['smooth_x']) - np.size(twop_dict['s2p_spks'], 1)

        # headx = np.array([np.mean([rearx[f], learx[f]]) for f in range(len(rearx))])
        # heady = np.array([np.mean([reary[f], leary[f]]) for f in range(len(reary))])
        sc = fm2p.SpatialCoding(cfg)
        sc.add_data(
            top_tracking_dict,
            arena_dict,
            dFF_transients,
            normspikes
        )

        print("Calculating place cells for recording {}".format(full_rname))
        occupancy_map, activity_maps = sc.calc_place_cells()

        print("reliability check for recording {}".format(full_rname))
        place_cell_inds, criteria_dict = sc.check_place_cell_reliability()

        criteria_dict['place_cell_inds'] = place_cell_inds
        cohens_d = criteria_dict['cohens_d']
        sigReliability = criteria_dict['place_cell_reliability']
        
        preprocessed_dict = {
            **top_tracking_dict,
            **top_xyl.to_dict(),
            **arena_dict,
            **twop_dict,
            **criteria_dict
        }

        # Add the activity maps separately
        preprocessed_dict['activity_maps'] = activity_maps
        preprocessed_dict['occupancy_map'] = occupancy_map
        
        # check if a folder exists (rpath\\place_cells) and if not, create it
        place_cells_dir = os.path.join(rpath, 'place_cells')
        if not os.path.exists(place_cells_dir):
            os.makedirs(place_cells_dir)
                    
        print("Plotting place cell maps for {}".format(full_rname))
        place_cell_indices = np.where(place_cell_inds)[0] 
        sigRel_indices = np.where(sigReliability)[0]
        # find indices that are false in criteria_dict['place_cell_reliability']
        plot_place_cell_maps(place_cell_indices, activity_maps, cohens_d, cohens_thresh, place_cells_dir, multi_criteria = True, sigma=1)
        plot_place_cell_maps(sigRel_indices, activity_maps, cohens_d, cohens_thresh, place_cells_dir, multi_criteria = False, sigma=1)
        
        from scipy.ndimage import gaussian_filter
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        def plot_place_unreliablecell_maps(cellIndices, activity_maps, cohens_d,savedir, multi_criteria = True, sigma=1):
            # sigma is std of gaussian filter

            if multi_criteria: 
                pdf = PdfPages(os.path.join(savedir, 'unreliable_place_cell_maps_multiCriteria{}_cohensthresh_{}.pdf').format(cohens_thresh, fm2p.fmt_now(c=True)))
            else:
                pdf = PdfPages(os.path.join(savedir, 'unreliable_place_cell_maps_cohensDonly{}_cohensthresh_{}.pdf').format(cohens_thresh, fm2p.fmt_now(c=True)))

            panel_width = 4
            panel_height = 5

            # valid_PCs is a boolean array; get indices of True values
            nPlaceCells = len(cellIndices)

            for batchStart in range(0, nPlaceCells, panel_width*panel_height):
                batchEnd = min(batchStart + panel_width*panel_height, nPlaceCells)

                fig, axs = plt.subplots(panel_width, panel_height, figsize=(15, 10))
                axs = axs.flatten()

                for i, ax in enumerate(axs[:batchEnd - batchStart]):

                    cell_idx = cellIndices[batchStart + i]
                    activity_map = activity_maps[cell_idx, :, :]
                    
                    # Replace NaN with 0 before smoothing
                    activity_map_clean = np.nan_to_num(activity_map, nan=0.0)

                    smoothedMap = gaussian_filter(activity_map_clean, sigma=sigma)

                    im = ax.imshow(smoothedMap.T, cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f'Cell {cell_idx}, Cohen\'s d: {cohens_d[cell_idx]:.2f}')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Hide any unused subplots
                for j in range(batchEnd - batchStart, len(axs)):
                    axs[j].axis('off')

                fig.suptitle(f'Unreiliable place Cells {cellIndices[batchStart]}–{cellIndices[batchEnd - 1]} of {nPlaceCells}')
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)

                pdf.savefig()
                plt.close(fig)

            pdf.close()
            
        unreliable_place_cells_indices = np.where(~place_cell_inds)[0]
        unreliable_indices = np.where(~sigReliability)[0]
        plot_place_unreliablecell_maps(unreliable_place_cells_indices, activity_maps, cohens_d, place_cells_dir, multi_criteria= True, sigma=1)
        plot_place_unreliablecell_maps(unreliable_indices, activity_maps, cohens_d, place_cells_dir, multi_criteria= False, sigma=1)

        
    
        # plot_place_cell_maps(place_cell_inds, activity_maps, savedir, sigma=1)
        _savepath = os.path.join(rpath, '{}_preproc_withActivityMap_cohensthresh_{}.h5'.format(rname, cohens_thresh))
        print('Writing preprocessed data to {}'.format(_savepath))
        fm2p.write_h5(_savepath, preprocessed_dict)


if __name__ == "__main__":
    # Replace 'your_config_file_path.yaml' with the actual path to your config file
    cfg_path = r'D:\pregnancy\250701_NSW130_Baseline3\config_HP.yaml'
    hippocampal_preprocess(cfg_path)      