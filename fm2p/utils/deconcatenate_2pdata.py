import h5py
import fm2p
import numpy as np
import os
import fnmatch
import cv2
import pandas as pd

from fm2p.utils.trim_video_opencv import trim_video_opencv

def find_folders(pattern, path, type, MR=False):
    """ Glob for subdirectories.

    Parameters
    ----------
    pattern : str
        Pattern to search for within the given `path`. Use
        a '*' for missing sections of characters.
    path : str
        Path to search, including subdirectories.
    MR : bool
        If only the most recent file should be returned. When
        MR is True, `_ret` will be returned as a str and be
        the path for the file matching the pattern which was
        written most recently. Other files which match the
        pattern but are not the most recent will be ignored.
        When MR is False, `_ret` is returned as a list of all
        files that matched the pattern. Default for MR is False.
    
    Returns
    -------
    _ret : list or str
        When MR is False, `_ret` is a list of files matching
        pattern. Otherwise when MR is True, `_ret` is a str
        containing only the path to the file which matched the
        pattern and was written most recently.
    """

    result = []
        
    # Walk through the path directory
    for root, dirs, files in os.walk(path):
        
        if type == 'folders':
            # Search through directories
            for name in dirs:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root))
        elif type == 'files':
            # Search through files
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root))

    if MR is True:
        # Return only the most recent result
        if result:
            _ret = max(result, key=os.path.getmtime)
        else:
            _ret = None
            
    elif MR is False:
        # Return the full list of items matching the pattern
        _ret = result

    return _ret



def split_and_save_video_opencv(video_path, frame_ranges, recording_dirs):
    """Split video using OpenCV instead of moviepy"""
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # print(f"Video info: {fps} FPS, {width}x{height}")
    
    for i, ((start_frame, end_frame), recording_dir) in enumerate(zip(frame_ranges, recording_dirs)):
        # Set up output video writer
        output_path = os.path.join(recording_dir, f'topdown_video_recording_{i+1}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Go to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Write frames from start_frame to end_frame
        frames_written = 0
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_num}")
                break
            out.write(frame)
            frames_written += 1
        
        out.release()
        print(f"Saved video segment {i+1}: frames {start_frame}-{end_frame} ({frames_written} frames) to {output_path}")
    
    cap.release()
    
def get_frame_ranges_from_user():
    """
    Get frame start and end numbers from user input for 4 recordings.
    
    Returns:
    --------
    frame_ranges : list
        List of 4 tuples (start_frame, end_frame) for each recording
    """
    print("Please enter the start and end frame numbers for each of the 4 recordings:")
    frame_ranges = []
    
    for i in range(4):
        while True:
            try:
                start_frame = int(input(f"Start frame for recording {i+1}: "))
                end_frame = int(input(f"End frame for recording {i+1}: "))
                
                if start_frame < 0 or end_frame < 0:
                    print("Frame numbers must be non-negative. Please try again.")
                    continue
                if end_frame <= start_frame:
                    print("End frame must be greater than start frame. Please try again.")
                    continue
                    
                frame_ranges.append((start_frame, end_frame))
                print(f"Recording {i+1}: frames {start_frame} to {end_frame-1} ({end_frame-start_frame} frames)")
                break
            except ValueError:
                print("Please enter valid integers.")
    
    return frame_ranges


def deconcatenate_continuous_data(rpathtwoup, cfg_path, frame_ranges):
    """
    Deconcatenate continuous DLC and Suite2p data based on provided frame starts.
    
    Parameters:
    -----------
    rpathtwoup : str
        Path to the main data directory
    cfg_path : str
        Path to the configuration file
    frame_starts : list
        List of start frame numbers for each recording
        
    Returns:
    --------
    twopdata : list
        A list of dictionaries containing deconcatenated 2P data for each recording.
    """
    
    cfg = fm2p.read_yaml(cfg_path)
    
    
    # Topdown camera files
    possible_topdown_videos = fm2p.find('*.mp4', rpathtwoup, MR=False)
    topdown_video = fm2p.filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)
    topdown_input_path = topdown_video
        
    # Find the single suite2p folder
    suite2p_dir = os.path.join(rpathtwoup, 'suite2p')
    if not os.path.exists(suite2p_dir):
        raise ValueError("Expected 1 suite2p folder, found none")
    
    # Load Suite2p data
    F_path = fm2p.find('F.npy', suite2p_dir, MR=True)
    Fneu_path = fm2p.find('Fneu.npy', suite2p_dir, MR=True)
    suite2p_spikes = fm2p.find('spks.npy', suite2p_dir, MR=True)
    iscell_path = fm2p.find('iscell.npy', suite2p_dir, MR=True)
    stat_path = fm2p.find('stat.npy', suite2p_dir, MR=True)
    ops_path = fm2p.find('ops.npy', suite2p_dir, MR=True)

    F = np.load(F_path, allow_pickle=True)
    Fneu = np.load(Fneu_path, allow_pickle=True)
    spks = np.load(suite2p_spikes, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)
    stat = np.load(stat_path, allow_pickle=True)
    ops = np.load(ops_path, allow_pickle=True)
    
    # Load continuous DLC data
    topdown_pts_path = fm2p.find('*DLC_resnet50_*preg_mini2p*.h5', rpathtwoup, MR=True)
    if topdown_pts_path is None:
        raise ValueError("Could not find continuous DLC file")
    topdown_video_path = fm2p.find('*labeled.mp4', rpathtwoup, MR=True)

    # Get video FPS
    if topdown_video_path:
        cap = cv2.VideoCapture(topdown_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # print(f"Video FPS: {fps}")
        
    xyl, _ = fm2p.open_dlc_h5(topdown_pts_path)
    total_dlc_frames = np.size(xyl, 0)
    total_2p_frames = np.size(F, 1)
    
    print(f"Total DLC frames: {total_dlc_frames}")
    print(f"Total 2P frames: {total_2p_frames}")
    
    
    # Initialize twopdata and dlcdata list
    twopdata = []
    for r in range(len(frame_ranges)):
        twopdata.append({})
    #     dlcdata.append({})
    
    # Create DLC-style MultiIndex columns
    # DLC format is typically: ('scorer', 'bodypart', 'coords')
    scorer = 'DLC_resnet50_preg_mini2p'  # Use your actual scorer name
    bodyparts = ['nose', 'nose', 'nose', 'back', 'back', 'back']
    coords = ['x', 'y', 'likelihood', 'x', 'y', 'likelihood']
    # Create MultiIndex
    multi_columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp, coord in zip(bodyparts, coords)],
        names=['scorer', 'bodyparts', 'coords']
    )
    # Deconcatenate and save data
    for r in range(len(frame_ranges)):
        start_frame, end_frame = frame_ranges[r]
        
        # Extract data for this recording
        twopdata[r]['F'] = F[:, start_frame:end_frame]
        twopdata[r]['Fneu'] = Fneu[:, start_frame:end_frame]
        twopdata[r]['spks'] = spks[:, start_frame:end_frame]
        twopdata[r]['iscell'] = iscell
        twopdata[r]['stat'] = stat
        twopdata[r]['ops'] = ops
        
        # Extract DLC data for this recording (already loaded in all_dlc_data[r])
        dlc_segment = xyl.iloc[start_frame:end_frame, :]


        # create a folder called suite2p inside the rpath for each recording
        individual_s2p_dir = os.path.join(rpathtwoup, 'Processed',str(r+1), 'suite2p')
        if not os.path.exists(individual_s2p_dir):
            os.makedirs(individual_s2p_dir)
            
        # Save Suite2p data
        np.save(os.path.join(individual_s2p_dir, 'F.npy'), twopdata[r]['F'])
        np.save(os.path.join(individual_s2p_dir, 'Fneu.npy'), twopdata[r]['Fneu'])
        np.save(os.path.join(individual_s2p_dir, 'spks.npy'), twopdata[r]['spks'])
        np.save(os.path.join(individual_s2p_dir, 'iscell.npy'), twopdata[r]['iscell'])
        np.save(os.path.join(individual_s2p_dir, 'stat.npy'), twopdata[r]['stat'])
        np.save(os.path.join(individual_s2p_dir, 'ops.npy'), twopdata[r]['ops'])
        
                
        # Set MultiIndex columns
        dlc_segment.columns = multi_columns
        print("MultiIndex columns:", dlc_segment.columns)

        # Save DLC data segment (you may need to adjust this based on your fm2p library)
        dlc_output_path = os.path.join(os.path.dirname(individual_s2p_dir), f'splitted_DLC_recording_{r+1}.h5')
        dlc_segment.to_hdf(dlc_output_path, key='df_with_missing', mode='w')
        
        # Extract top down video
        topdown_output_path = os.path.join(os.path.dirname(individual_s2p_dir), f'topdownvideo_{r+1}.mp4')
        num_frames = end_frame - start_frame  # Number of frames to keep
        fps = 7.52  # Optional: specify output framerate

        trim_video_opencv(topdown_input_path, topdown_output_path, start_frame, num_frames, fps)
        
        # Test
        test_load, test_names = fm2p.open_dlc_h5(dlc_output_path)
        print(f"Final columns: {test_load.columns.tolist()}")
        
        # if topdown_video_path:
        #     split_and_save_video_opencv(topdown_video_path, [frame_ranges[r]], [os.path.dirname(individual_s2p_dir)])
        # else:
        #     print("No video file found to split")
        print(f"Saved recording {r+1} data to {os.path.dirname(individual_s2p_dir)}")
        print(f"  Suite2p frames: {twopdata[r]['F'].shape[1]}")
        print(f"  DLC frames: {dlc_segment.shape[0]}")
        
    return twopdata


def deconcatenate_twopdata(rpathtwoup, cfg_path, frame_ranges=None):
    """
    Deconcatenate 2P data from multiple recordings into separate data files.
    Automatically detects data type based on DLC file structure.
        
    Parameters:
    -----------
    rpathtwoup : str
        Path to the main data directory
    cfg_path : str  
        Path to the configuration file
    frame_ranges : list, optional
        List of (start_frame, end_frame) tuples for continuous data (Type 2 only)
        
    Returns:
    --------
    twopdata : list
        A list of dictionaries containing deconcatenated 2P data for each recording.
    """
    
    if rpathtwoup is None:
        rpathtwoup = r"F:\2P\pregnancy\2p_data\250701_NSW130_Baseline3"
    if cfg_path is None:
        cfg_path = r"F:\2P\pregnancy\2p_data\250701_NSW130_Baseline3\config_HP.yaml"
    
    # Check for DLC files to determine data type
    # Type 1: Multiple DLC files (one in each recording folder)
    # Type 2: Single continuous DLC file (in main directory)
    
    # Look for DLC files in main directory (Type 2 - continuous)
    dlc_files = fm2p.find('*DLC_resnet50_*preg_mini2p*.h5', rpathtwoup, MR=False)
    
    print(f"Found {len(dlc_files)} DLC file(s) in recording folders")
    
    if len(dlc_files) == 4:
        print("Data Type 1: Separate DLC recordings + concatenated Suite2p")
        print("Will deconcatenate Suite2p based on DLC frame counts...")
        return deconcatenate_separate_recordings(rpathtwoup, cfg_path)
        
    elif len(dlc_files) == 1:
        print("Data Type 2: Continuous recordings (single DLC + single Suite2p)")
        if frame_ranges is None:
            frame_ranges = get_frame_ranges_from_user()
        print("Will deconcatenate both DLC and Suite2p using provided frame ranges...")
        return deconcatenate_continuous_data(rpathtwoup, cfg_path, frame_ranges)
        
    else:
        raise ValueError(f"Ambiguous data structure: {len(dlc_files)} DLC files in main directory, {dlc_files} in recording folders. Expected either separate DLC files OR single continuous DLC file.")


def deconcatenate_separate_recordings(rpathtwoup, cfg_path):
    """
    Original function for Type 1 data: separate DLC recordings + concatenated Suite2p
    """
    rpath = []
    cfg = fm2p.read_yaml(cfg_path)

    dlcpath = fm2p.find('*DLC_resnet50_*preg_mini2p*.h5', rpathtwoup, MR=False)
    possible_topdown_videos = fm2p.find('*labeled.mp4', rpathtwoup, MR=False)

    numDLCframes = np.zeros(np.size(dlcpath), dtype=int) 
    for r in range(np.size(possible_topdown_videos)):
        # Topdown camera files
        # topdown_video = fm2p.filter_file_search(possible_topdown_videos[r], toss=[], MR=True)
        
        top_cam = fm2p.Topcam(rpath, '', cfg=cfg)

        topdown_video = possible_topdown_videos[r]
        topdown_pts_path = dlcpath[r]

        top_cam.add_files(
                    top_dlc_h5=topdown_pts_path,
                    top_avi=topdown_video
                )
        xyl, _ = fm2p.open_dlc_h5(top_cam.top_dlc_h5)

        numDLCframes[r] = np.size(xyl,0)
        
    suite2p_dir = os.path.join(rpathtwoup, "suite2p")
    
    F_path = fm2p.find('F.npy', suite2p_dir, MR=True)
    Fneu_path = fm2p.find('Fneu.npy', suite2p_dir, MR=True)
    suite2p_spikes = fm2p.find('spks.npy', suite2p_dir, MR=True)
    iscell_path = fm2p.find('iscell.npy', suite2p_dir, MR=True)
    stat_path = fm2p.find('stat.npy', suite2p_dir, MR=True)
    ops_path = fm2p.find('ops.npy', suite2p_dir, MR=True)

    F = np.load(F_path, allow_pickle=True)
    Fneu = np.load(Fneu_path, allow_pickle=True)
    spks = np.load(suite2p_spikes, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)
    stat = np.load(stat_path, allow_pickle=True)
    ops =  np.load(ops_path, allow_pickle=True)
    numtwopframes = np.shape(F)[1]
    twopdata = []
    

    # Initialize the list with dictionaries for each recording
    for r in range(np.size(dlcpath,0)):
        twopdata.append({})

    if numtwopframes == np.sum(numDLCframes):
        # Calculate frame ranges for each recording
        frame_start = 0
        for r in range(np.size(dlcpath,0)):
            frame_end = frame_start + numDLCframes[r]
            twopdata[r]['F'] = F[:, frame_start:frame_end]
            twopdata[r]['Fneu'] = Fneu[:, frame_start:frame_end]
            twopdata[r]['spks'] = spks[:, frame_start:frame_end]    
            twopdata[r]['iscell'] = iscell
            twopdata[r]['stat'] = stat
            twopdata[r]['ops'] = ops

            # create a folder called suite2p inside the rpath for each recording
            if not os.path.exists(os.path.join(rpathtwoup, 'Processed',str(r+1), 'suite2p')):
                os.makedirs(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p'))

            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'F.npy'), twopdata[r]['F'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'Fneu.npy'), twopdata[r]['Fneu'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'spks.npy'), twopdata[r]['spks'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'iscell.npy'), twopdata[r]['iscell'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'stat.npy'), twopdata[r]['stat'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'ops.npy'), twopdata[r]['ops'])
            
            frame_start = frame_end
    elif numtwopframes != np.sum(numDLCframes) and np.abs(numtwopframes - np.sum(numDLCframes)) == 4:
        frame_start = 0
        for r in range(np.size(dlcpath,0)):
            if numtwopframes < np.sum(numDLCframes):
                frame_end = frame_start + numDLCframes[r]
            else:
                frame_end = frame_start + numDLCframes[r]+1
                
            twopdata[r]['F'] = F[:, frame_start:frame_end]
            twopdata[r]['Fneu'] = Fneu[:, frame_start:frame_end]
            twopdata[r]['spks'] = spks[:, frame_start:frame_end]    
            twopdata[r]['iscell'] = iscell
            twopdata[r]['stat'] = stat
            twopdata[r]['ops'] = ops
            
            # create a folder called suite2p inside the rpath for each recording
            if not os.path.exists(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p')):
                os.makedirs(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p'))

            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'F.npy'), twopdata[r]['F'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'Fneu.npy'), twopdata[r]['Fneu'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'spks.npy'), twopdata[r]['spks'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'iscell.npy'), twopdata[r]['iscell'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'stat.npy'), twopdata[r]['stat'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'ops.npy'), twopdata[r]['ops'])
            
            frame_start = frame_end + 1
    elif numtwopframes != np.sum(numDLCframes) and np.abs(numtwopframes - np.sum(numDLCframes)) == 2:
        for r in range(np.size(dlcpath,0)):
            if numtwopframes - np.sum(numDLCframes) == -2:
                if r == 0:
                    frame_start = 0
                    frame_end = numDLCframes[0]
                    print(f"frame start {frame_start}, frame end {frame_end}")
                elif r == 1:
                    frame_start = numDLCframes[0]
                    frame_end = frame_start+numDLCframes[1]-1
                    print(f"frame start {frame_start}, frame end {frame_end}")

                elif r == 2:
                    frame_start = numDLCframes[0]+numDLCframes[1]
                    frame_end = frame_start + numDLCframes[2]
                    print(f"frame start {frame_start}, frame end {frame_end}")

                elif r == 3:
                    frame_start = numDLCframes[0]+numDLCframes[1] + numDLCframes[2]
                    frame_end = frame_start+numDLCframes[3]-1
                    print(f"frame start {frame_start}, frame end {frame_end}")
            elif numtwopframes - np.sum(numDLCframes) == 2:
                if r == 0:
                    frame_start = 0
                    frame_end = numDLCframes[0]
                    print(f"frame start {frame_start}, frame end {frame_end}")
                elif r == 1:
                    frame_start = numDLCframes[0]
                    frame_end = frame_start+numDLCframes[1]+1
                    print(f"frame start {frame_start}, frame end {frame_end}")

                elif r == 2:
                    frame_start = numDLCframes[0]+numDLCframes[1]+1
                    frame_end = frame_start + numDLCframes[2]
                    print(f"frame start {frame_start}, frame end {frame_end}")

                elif r == 3:
                    frame_start = numDLCframes[0]+numDLCframes[1] + numDLCframes[2]+1
                    frame_end = frame_start+numDLCframes[3]+1
                    print(f"frame start {frame_start}, frame end {frame_end}")

                
            twopdata[r]['F'] = F[:, frame_start:frame_end]
            twopdata[r]['Fneu'] = Fneu[:, frame_start:frame_end]
            twopdata[r]['spks'] = spks[:, frame_start:frame_end]    
            twopdata[r]['iscell'] = iscell
            twopdata[r]['stat'] = stat
            twopdata[r]['ops'] = ops
            
            # create a folder called suite2p inside the rpath for each recording
            if not os.path.exists(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p')):
                os.makedirs(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p'))

            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'F.npy'), twopdata[r]['F'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'Fneu.npy'), twopdata[r]['Fneu'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'spks.npy'), twopdata[r]['spks'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'iscell.npy'), twopdata[r]['iscell'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'stat.npy'), twopdata[r]['stat'])
            np.save(os.path.join(rpathtwoup, 'Processed', str(r+1), 'suite2p', 'ops.npy'), twopdata[r]['ops'])
            
            frame_start = frame_end + 1
                    
    elif numtwopframes != np.sum(numDLCframes) and (numtwopframes - np.sum(numDLCframes) < 4 or numtwopframes - np.sum(numDLCframes) < 0):
        # throw an error if the number of 2P frames does not match the number of DLC frames
        raise ValueError("Mismatch between number of 2P frames and DLC frames. Please check the data files.")

    return twopdata