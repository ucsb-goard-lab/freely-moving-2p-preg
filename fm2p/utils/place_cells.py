
import os
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from numpy import log2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages

import fm2p

class SpatialCoding():

    def __init__(self, cfg):

        self.cfg = cfg

        self.bin_size = cfg['place_bin_size'] # in cm
        self.sd_thresh = cfg['place_sd_thresh']
        # number of pixels between coordinates below which is considered 'not moving'
        self.move_thresh = cfg['running_thresh']
        self.likelihood_thresh = cfg['likelihood_thresh']

        self.cohens_thresh = cfg['cohens_d']
        self.nCells = 0
        self.x = None
        self.y = None
        self.spikes = None


    def add_data(self, topdown_dict, arena_dict, dFF_transients, normspikes):

        self.x = topdown_dict['smooth_x']
        self.y = topdown_dict['smooth_y']
        
        # Ensure that speed is the same length as position data
        # self.speed = topdown_dict['speed']
        self.speed = np.append(topdown_dict['speed'], topdown_dict['speed'][-1])
        self.useF = self.speed > self.move_thresh
        self.dFF_transients = dFF_transients
        self.normspikes = normspikes
        self.nCells = np.size(dFF_transients, 0)
        self.arena = arena_dict
        
        
    def calc_place_cells(self):

        assert self.nCells > 0
        assert self.dFF_transients is not None
        assert self.x is not None
        assert self.y is not None
        assert self.normspikes is not None
        assert self.arena is not None
        
        dFF_transients = self.dFF_transients.copy()[:,self.useF]
        normspikes = self.normspikes.copy()[:,self.useF]
        
        x = self.x.copy()[self.useF]
        y = self.y.copy()[self.useF]

        # bin size in units of pixels
        bin_size_pxls = self.bin_size * self.arena['pxls2cm']
        # bin_size_pxls = 40
        x_edges = np.linspace(
            np.floor(np.min(x)),
            np.ceil(np.max(x)),
            num=int((np.ceil(np.max(x))-np.floor(np.min(x))) / bin_size_pxls)
        )
        y_edges = np.linspace(
            np.floor(np.min(y)),
            np.ceil(np.max(y)),
            num=int((np.ceil(np.max(y))-np.floor(np.min(y))) / bin_size_pxls)
        )
        num_bins_x = len(x_edges) - 1
        num_bins_y = len(y_edges) - 1

        # hist of occupancy
        occupancy_map, occ_x, occ_y = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges]
        )

        activity_maps = np.zeros([
            self.nCells,
            occupancy_map.shape[0],  # Use actual histogram dimensions
            occupancy_map.shape[1]   # instead of bin edge dimensions
        ])

        for c in range(self.nCells):
            actmap_,_,_ = np.histogram2d(
                x,
                y,
                bins=[x_edges, y_edges],
                weights=dFF_transients[c, :]
            )

            # avoid dividing by zero
            occupancy_map[occupancy_map == 0] = np.nan

            activity_maps[c,:,:] = actmap_ / occupancy_map

        self.occupancy_map = occupancy_map
        self.activity_maps = activity_maps
        
        #  # DEBUG: Print spatial analysis info
        # print(f"\n=== Spatial Analysis Debug ===")
        # print(f"Position range: x=({np.min(x):.1f}, {np.max(x):.1f}), y=({np.min(y):.1f}, {np.max(y):.1f})")
        # print(f"Bin size: {self.bin_size} cm = {bin_size_pxls:.1f} pixels")
        # print(f"Number of bins: {num_bins_x} x {num_bins_y} = {num_bins_x * num_bins_y}")
        # print(f"Activity maps shape: {activity_maps.shape}")
        # print(f"Frames used for analysis: {np.sum(self.useF)} / {len(self.useF)}")
        # print(f"pxls2cm factor: {self.arena['pxls2cm']:.2f}")
    
        return occupancy_map, activity_maps
          

    def check_place_cell_reliability(self, dFF_transients=None, x=None, y=None):
        
        if dFF_transients is None:
            dFF_transients = self.dFF_transients.copy()[:,self.useF]
        if x is None:
            x = self.x.copy()[self.useF]
        if y is None:
            y = self.y.copy()[self.useF]
        
        cohens_d = self.cfg['cohens_d']
        bout_duration = self.cfg['bout_duration']
        nShuffles = self.cfg['n_pc_shuffles']
        bin_size = self.bin_size * self.arena['pxls2cm']  # convert to pixels
        n_bouts = self.cfg['n_bouts']

        nCells, nFrames = np.shape(dFF_transients)

        if len(x) != nFrames or len(y) != nFrames:
            print(f"Mismatch: dFF_transients has {nFrames} frames, but x has {len(x)} and y has {len(y)} points")
            # # You'll need to handle this - either trim the longer one or interpolate the shorter one
            # min_length = min(nFrames, len(x), len(y))
            # x = x[:min_length]
            # y = y[:min_length]
            # dFF_transients = dFF_transients[:, :min_length]
            # nFrames = min_length
            
        xEdges = np.arange(np.floor(x.min()), np.ceil(x.max()) + bin_size, bin_size)
        yEdges = np.arange(np.floor(y.min()), np.ceil(y.max()) + bin_size, bin_size)
        nBinsX = len(xEdges) - 1
        nBinsY = len(yEdges) - 1
        nBins = nBinsX * nBinsY

        xBin = np.digitize(x, xEdges) - 1
        yBin = np.digitize(y, yEdges) - 1

        # occupancy
        valid = (xBin >= 0) & (yBin >= 0) & (xBin < nBinsX) & (yBin < nBinsY)
        occupancyMap = np.zeros((nBinsY, nBinsX))
        for xb, yb in zip(xBin[valid], yBin[valid]):
            occupancyMap[yb, xb] += 1
        occupancyFlat = occupancyMap.flatten()
        p_i = occupancyFlat / np.sum(occupancyFlat) # percentage of occupancy in each bin

        # bin index per frame
        binIdx = np.zeros(nFrames, dtype=int)
        for i in range(nFrames):
            # if 0 <= xBin[i] < nBinsX and 0 <= yBin[i] < nBinsY:
            if xBin[i] > 0 and yBin[i] > 0:
                binIdx[i] = yBin[i] * nBinsX + xBin[i]
            # else:
            #     binIdx[i] = -1  # invalid bin
                
                
        # # bin index per frame
        # binIdx = np.zeros(nFrames, dtype=int)
        # for i in range(nFrames):
        #     if 0 <= xBin[i] < nBinsX and 0 <= yBin[i] < nBinsY:
        #         binIdx[i] = yBin[i] * nBinsX + xBin[i]
        #     else:
        #         binIdx[i] = -1  # invalid bin

        # spatial information
        activityFlat = np.zeros((nBins, nCells))
        for c in range(nCells):
            r_i = np.zeros(nBins)
            for b in range(nBins):
                valid_idx = (binIdx == b)
                if occupancyFlat[b] > 0:
                    r_i[b] = np.sum(dFF_transients[c, valid_idx]) / occupancyFlat[b]
            activityFlat[:, c] = r_i # how much each cell responds in each spatial bin

        spatialInfo = np.zeros(nCells)
        for c in range(nCells):
            r_i = activityFlat[:, c]
            r_i[r_i == 0] = np.finfo(float).eps
            r_bar = np.sum(p_i * r_i)
            spatialInfo[c] = np.sum(p_i * (r_i / r_bar) * np.log2(r_i / r_bar)) # how good each cell is as a place cell

        # Shuffled SI
        shuffledSI = np.zeros((nShuffles, nCells))
        for s in range(nShuffles):
            for c in range(nCells):
                shuffled_trace = np.roll(dFF_transients[c,:], np.random.randint(nFrames))
                r_i = np.zeros(nBins)
                for b in range(nBins):
                    valid_idx = (binIdx == b)
                    if occupancyFlat[b] > 0:
                        r_i[b] = np.sum(shuffled_trace[valid_idx]) / occupancyFlat[b]
                r_i[r_i == 0] = np.finfo(float).eps
                r_bar = np.sum(p_i * r_i)
                shuffledSI[s, c] = np.sum(p_i * (r_i / r_bar) * np.log2(r_i / r_bar))

        sigSI = spatialInfo > np.percentile(shuffledSI, 85, axis=0)

        # Consistency via Cohen's d
        reliability = np.zeros(nCells)
        dFF_transients_rand = np.zeros_like(dFF_transients)
        
        for c in range(nCells):
            print(f"Calculating reliability for cell {c+1}/{nCells}")
            bt_CC_data = np.zeros(nShuffles)
            bt_CC_rand = np.zeros(nShuffles)
            
            for shuffles in range(nShuffles):
                # for raw data
                idxA_start = np.zeros(n_bouts, dtype=int)
                idxB_start = np.zeros(n_bouts, dtype=int)

                idxA = np.zeros((n_bouts, bout_duration), dtype=int)
                idxB = np.zeros((n_bouts, bout_duration), dtype=int)

                aVals = np.zeros((n_bouts, bout_duration), dtype=float)
                bVals = np.zeros((n_bouts, bout_duration), dtype=float)

                aAct = np.zeros((n_bouts, nBins), dtype=float)
                bAct = np.zeros((n_bouts, nBins), dtype=float)
                
                for bouts in range(n_bouts):
                    idxA_start[bouts] = np.random.randint(nFrames - bout_duration + 1)
                    idxB_start[bouts] = np.random.randint(nFrames - bout_duration + 1)
                    idxA[bouts] = np.arange(idxA_start[bouts], idxA_start[bouts] + bout_duration)
                    idxB[bouts] = np.arange(idxB_start[bouts], idxB_start[bouts] + bout_duration)
                
                    aBins = binIdx[idxA[bouts]]
                    bBins = binIdx[idxB[bouts]]
                    aVals[bouts,:] = dFF_transients[c, idxA[bouts]]
                    bVals[bouts,:] = dFF_transients[c, idxB[bouts]]

                    for b in range(nBins):
                        aMask = aBins == b
                        bMask = bBins == b
                        if np.any(aMask):
                            aAct[bouts,b] = np.sum(aVals[bouts, aMask])
                        if np.any(bMask):
                            bAct[bouts,b] = np.sum(bVals[bouts, bMask])
                            
                            
                CC = np.corrcoef(np.mean(aAct, axis=0), np.mean(bAct, axis=0))[0, 1]
                bt_CC_data[shuffles] = CC

                
            for shuffles in range(nShuffles):
                # for circularly shifted data
                idxA_start = np.zeros(n_bouts, dtype=int)
                idxB_start = np.zeros(n_bouts, dtype=int)

                idxA = np.zeros((n_bouts, bout_duration), dtype=int)
                idxB = np.zeros((n_bouts, bout_duration), dtype=int)

                aVals = np.zeros((n_bouts, bout_duration), dtype=float)
                bVals = np.zeros((n_bouts, bout_duration), dtype=float)

                aAct = np.zeros((n_bouts, nBins), dtype=float)
                bAct = np.zeros((n_bouts, nBins), dtype=float)
                
                dFF_transients_rand = np.roll(dFF_transients, np.random.randint(nFrames), axis=1)

                for bouts in range(n_bouts):
                    # A = original data, B = circularly shifted data
                    idxA_start[bouts] = np.random.randint(nFrames - bout_duration + 1)
                    idxB_start[bouts] = np.random.randint(nFrames - bout_duration + 1)
                    idxA[bouts] = np.arange(idxA_start[bouts], idxA_start[bouts] + bout_duration)
                    idxB[bouts] = np.arange(idxB_start[bouts], idxB_start[bouts] + bout_duration)
                
                    aBins = binIdx[idxA[bouts]]
                    bBins = binIdx[idxB[bouts]]
                    aVals[bouts,:] = dFF_transients[c, idxA[bouts]]
                    bVals[bouts,:] = dFF_transients_rand[c, idxB[bouts]]

                    for b in range(nBins):
                        aMask = aBins == b
                        bMask = bBins == b
                        if np.any(aMask):
                            aAct[bouts,b] = np.sum(aVals[bouts, aMask])
                        if np.any(bMask):
                            bAct[bouts,b] = np.sum(bVals[bouts, bMask])
                            
                            
                CC = np.corrcoef(np.mean(aAct, axis=0), np.mean(bAct, axis=0))[0, 1]
                bt_CC_rand[shuffles] = CC
    
            diff_mean = np.mean(bt_CC_data) - np.mean(bt_CC_rand)
            n1 = np.size(bt_CC_data)
            n2 = np.size(bt_CC_rand)
            var_x1 = np.var(bt_CC_data, ddof=1)  # sample variance
            var_x2 = np.var(bt_CC_rand, ddof=1)
            sv1 = ((n1-1)*var_x1)
            sv2 = ((n2-1)*var_x2)
            numer =  sv1 + sv2
            denom = (n1 + n2 - 2)
            pooled_std =  np.sqrt(numer / denom)
            d_value = diff_mean / pooled_std
            reliability[c] = d_value

        sigRel = reliability > cohens_d

        # place field contiguity
        hasPlaceField = np.zeros(nCells, dtype=bool)
        thresholdFrac = 0.4

        for c in range(nCells):
            rMap = activityFlat[:, c].reshape((nBinsY, nBinsX))
            rThresh = np.mean(rMap) * (1 + thresholdFrac)
            above = rMap > rThresh

            for i in range(nBinsY - 1):
                for j in range(nBinsX - 1):
                    block = above[i:i+2, j:j+2]
                    if np.all(block):
                        hasPlaceField[c] = True
                        break
                if hasPlaceField[c]:
                    break


        criteria_dict = {
            'place_cell_spatial_info': sigSI,
            'place_cell_reliability': sigRel,
            'cohens_d': reliability,
            'has_place_field': hasPlaceField
        }
        place_cell_inds = sigSI & sigRel & hasPlaceField
        print(f'Identified {np.sum(place_cell_inds)} place cells out of {nCells}.')

        self.place_cell_inds = place_cell_inds
        self.criteria_dict = criteria_dict

        return place_cell_inds, criteria_dict

def plot_place_cell_maps(cellIndices, activity_maps, cohens_d, cohens_thresh,savedir, multi_criteria = True,sigma=1):
    # sigma is std of gaussian filter

    if multi_criteria: 
        pdf = PdfPages(os.path.join(savedir, 'place_cell_maps_multiCriteria{}_cohensthresh_{}.pdf').format(cohens_thresh, fm2p.fmt_now(c=True)))
    else:
        pdf = PdfPages(os.path.join(savedir, 'place_cell_maps_cohensDonly{}_cohensthresh_{}.pdf').format(cohens_thresh, fm2p.fmt_now(c=True)))
    
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

        fig.suptitle(f'Place Cells {cellIndices[batchStart]}–{cellIndices[batchEnd - 1]} of {nPlaceCells}')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        pdf.savefig()
        plt.close(fig)

    pdf.close()

        


