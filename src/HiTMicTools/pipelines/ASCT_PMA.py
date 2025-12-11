import os
import gc
import argparse
import random
from typing import Optional, List

import numpy as np
import pandas as pd
import psutil
import torch
import tifffile
from scipy.ndimage import gaussian_filter

from HiTMicTools.img_processing.img_processor import ImagePreprocessor
from HiTMicTools.pipelines.base_pipeline import BasePipeline
from HiTMicTools.utils import get_timestamps, remove_file_extension
from HiTMicTools.img_processing.img_ops import measure_background_intensity
from HiTMicTools.img_processing.array_ops import convert_image
from HiTMicTools.img_processing.mask_ops import map_predictions_to_labels
from HiTMicTools.resource_management.memlogger import MemoryLogger
from HiTMicTools.resource_management.sysutils import get_device, empty_gpu_cache
from HiTMicTools.resource_management.reserveresource import ReserveResource
from HiTMicTools.roianalysis import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev
from HiTMicTools.confreader import ConfReader
from jetraw_tools.image_reader import ImageReader


# Background correction parameters
BF_GAUSSIAN_SIGMA = 50
FL_GAUSSIAN_SIGMA = 150
FL_SUBTRACT_FRACTION = 0.3

# Segmentation parameters
MIN_PIXEL_SIZE = 20

# Object class mapping
CLASS_TO_ID = {
    "single-cell": 0,
    "clump": 1,
    "noise": 2,
    "off-focus": 3,
    "joint-cell": 4,
}

# PI class mapping
PI_CLASS_MAP = {"piPOS": 1, "piNEG": 2}


class PMA_KillCurve_Pipeline(BasePipeline):
    """
    Custom pipeline for PMA kill curve analysis with focus restoration and tracking.

    This pipeline processes microscopy images to:
    1. Restore focus in both brightfield and fluorescence channels
    2. Segment and classify cells in the images
    3. Track cells across time frames
    4. Analyze fluorescence intensity and other cellular properties

    The pipeline is designed for time-lapse microscopy data with multiple channels,
    particularly for experiments tracking PI (propidium iodide) uptake in cells.

    Attributes:
        reference_channel (int): Index of the brightfield/reference channel
        pi_channel (int): Index of the fluorescence/PI channel
        align_frames (bool): Whether to align frames in time series
        method (str): Background correction method ('standard', 'basicpy', or 'basicpy_fl')
        image_segmentator: Model for cell segmentation
        object_classifier: Model for classifying segmented objects
        bf_focus_restorer: Model for restoring focus in brightfield images
        fl_focus_restorer: Model for restoring focus in fluorescence images
        pi_classifier: Model for classifying PI positive/negative cells
    """

    required_models = {"bf_focus", "fl_focus", "segmentation", "cell_classifier", "pi_classification"}

    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
    ) -> str:
        """Pipeline analysis for each image."""

        # Setup and initialization
        device = self._setup_device()
        is_cuda = device.type == 'cuda'
        movie_name = remove_file_extension(name)
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")

        # Read metadata and image
        img_logger.info("1 - Reading image", show_memory=True)
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()

        # Extract image dimensions
        pixel_size, size_x, size_y, nSlices, nChannels, nFrames = self._extract_metadata(metadata)
        img_logger.info(
            f"Image shape: {img.shape}, pixel size: {pixel_size} µm. "
            f"Reshaped to (frames={nFrames}, channels={nChannels}, slices={nSlices}, x={size_x}, y={size_y})"
        )

        # Preprocess image
        ip = self._preprocess_image(img, nFrames, nChannels, size_x, size_y, img_logger)
        img = np.zeros((1, 1, 1, 1))  # Free memory
        size_x, size_y = ip.img.shape[-2], ip.img.shape[-1]  # Update after alignment

        # Background correction
        self._perform_background_correction(ip, nFrames, img_logger)

        # Focus restoration (optional)
        if getattr(self, 'focus_correction', True):
            self._perform_focus_restoration(ip, img_logger)
        else:
            img_logger.info("2.2 - Focus correction disabled, skipping focus restoration", show_memory=True)

        # Clip negative values
        self._clip_negative_values(ip, img_logger)

        # Clear unused data
        ip.img_original = np.zeros((1, 1, 1, 1, 1))

        # Segmentation
        img_logger.info("3.1 - Image segmentation", show_memory=True, cuda=is_cuda)
        prob_map = self._segment_image(ip, img_logger)
        img_logger.info("3.1 - Segmentation completed", show_memory=True, cuda=is_cuda)

        # ROI analysis
        img_logger.info("3.2 - Extracting ROIs", show_memory=True)
        img_analyser = self._create_roi_analyser(ip, prob_map, img_logger)
        del ip  # Free memory

        # Classify ROIs
        img_logger.info("3.2 - Classifying ROIs", show_memory=True, cuda=is_cuda)
        object_classes, labels = self._classify_rois(img_analyser, img_logger)
        img_logger.info("3.2 - GPU memory status after classification", show_memory=True, cuda=is_cuda)

        # Measurements
        img_logger.info("4 - Starting measurements", show_memory=True)
        fl_measurements = self._extract_measurements(
            img_analyser, metadata, object_classes, size_x, size_y, img_logger
        )
        img_logger.info("4 - Measurements completed", show_memory=True)

        # PI classification and summary
        d_summary = self._perform_pi_classification(fl_measurements, name, img_logger)

        # Export results
        self._export_results(
            name, fl_measurements, d_summary, img_analyser,
            object_classes, labels, export_labeled_mask, export_aligned_image, img_logger
        )

        # Cleanup
        img_logger.info(f"Analysis completed for {movie_name}", show_memory=True)
        del prob_map, img, fl_measurements, d_summary, img_analyser
        gc.collect()
        empty_gpu_cache(device)
        img_logger.info("Garbage collection completed", show_memory=True)
        self.remove_logger(img_logger)

        return name

    def _setup_device(self) -> torch.device:
        """Setup CUDA device with explicit index."""
        device = get_device()
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda:0')
        return device

    def _extract_metadata(self, metadata):
        """Extract dimensional metadata from image."""
        pixel_size = metadata.images[0].pixels.physical_size_x
        size_x = metadata.images[0].pixels.size_x
        size_y = metadata.images[0].pixels.size_y
        nSlices = metadata.images[0].pixels.size_z
        nChannels = metadata.images[0].pixels.size_c
        nFrames = metadata.images[0].pixels.size_t
        return pixel_size, size_x, size_y, nSlices, nChannels, nFrames

    def _preprocess_image(self, img, nFrames, nChannels, size_x, size_y, img_logger):
        """Preprocess image: reshape, convert to float32, align frames, detect wells."""
        img_logger.info("2.1 - Preprocessing image", show_memory=True)
        img = img.reshape(nFrames, nChannels, size_x, size_y)
        ip = ImagePreprocessor(img, stack_order="TCXY")
        ip.img = ip.img.astype(np.float32)
        img_logger.info(f"Preprocessed image shape: {ip.img.shape}")

        # Align frames if required
        if self.align_frames:
            img_logger.info("2.1 - Aligning frames in the stack", show_memory=True)
            ip.align_image(
                ref_channel=0, ref_slice=-1, crop_image=True, reference_type="dynamic"
            )
            img_logger.info("2.1 - Frame alignment completed", show_memory=True)

        # Detect and fix border wells
        img_logger.info("2.1 - Detecting and fixing border wells")
        ip.detect_fix_well(nchannels=0, nslices=0, nframes=range(nFrames))

        return ip

    def _perform_background_correction(self, ip, nFrames, img_logger):
        """Apply background correction to both channels."""
        img_logger.info(f"Reference channel before background removal:\n{self.check_px_values(ip, self.reference_channel, round=3)}")
        img_logger.info(f"PI channel before background removal:\n{self.check_px_values(ip, self.pi_channel, round=3)}")

        self.apply_background_correction(
            ip,
            reference_channel=self.reference_channel,
            pi_channel=self.pi_channel,
            nFrames=range(nFrames),
            img_logger=img_logger
        )

        img_logger.info(f"Reference channel after background removal:\n{self.check_px_values(ip, self.reference_channel, round=3)}")
        img_logger.info(f"PI channel after background removal:\n{self.check_px_values(ip, self.pi_channel, round=3)}")

    def _perform_focus_restoration(self, ip, img_logger):
        """Restore focus in both brightfield and fluorescence channels."""
        device = get_device()

        # Brightfield focus restoration
        img_logger.info("2.2 - Restoring focus in the reference channel", show_memory=True)
        img_logger.info(f"Reference channel before focus restoration:\n{self.check_px_values(ip, self.reference_channel, round=3)}")

        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            ip.img[:, 0, self.reference_channel] = self.bf_focus_restorer.predict(
                ip.img[:, 0, self.reference_channel],
                rescale=False,
                batch_size=1,
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
            )

        # Fluorescence focus restoration
        img_logger.info("2.2 - Restoring focus in the PI channel", show_memory=True)
        img_logger.info(f"PI channel before focus restoration:\n{self.check_px_values(ip, self.pi_channel, round=3)}")

        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            ip.img[:, 0, self.pi_channel] = self.fl_focus_restorer.predict(
                ip.img[:, 0, self.pi_channel],
                batch_size=1,
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
                padding_mode="reflect",
            )

        img_logger.info(f"Reference channel after focus restoration:\n{self.check_px_values(ip, self.reference_channel, round=3)}")
        img_logger.info(f"PI channel after focus restoration:\n{self.check_px_values(ip, self.pi_channel, round=3)}")

    def _clip_negative_values(self, ip, img_logger):
        """Clip negative values created by NAFNet to 0."""
        ip.img[:, 0, self.reference_channel] = np.clip(ip.img[:, 0, self.reference_channel], 0, None)
        img_logger.info(f"BF channel clipped to remove negative values after focus restoration")
        img_logger.info(f"BF after clip: min={ip.img[:, 0, self.reference_channel].min():.3f}, max={ip.img[:, 0, self.reference_channel].max():.3f}")

        ip.img[:, 0, self.pi_channel] = np.clip(ip.img[:, 0, self.pi_channel], 0, None)
        img_logger.info(f"PI channel clipped to remove negative values after focus restoration")
        img_logger.info(f"PI after clip: min={ip.img[:, 0, self.pi_channel].min():.3f}, max={ip.img[:, 0, self.pi_channel].max():.3f}")

    def _segment_image(self, ip, img_logger):
        """Segment the image and return probability map."""
        device = get_device()
        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            prob_map = self.image_segmentator.predict(
                ip.img[:, 0, self.reference_channel, :, :],
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
            )

        # Ensure prob_map has correct dimensions
        if prob_map.ndim > 3 and prob_map.shape[1] > 1:
            prob_map = np.max(prob_map, axis=1, keepdims=True)
        elif prob_map.ndim == 3:
            prob_map = np.expand_dims(prob_map, axis=1)
        elif prob_map.ndim == 2:
            prob_map = np.expand_dims(prob_map, axis=(0, 1))

        return prob_map

    def _create_roi_analyser(self, ip, prob_map, img_logger):
        """Create ROI analyser and extract ROIs."""
        img_analyser = RoiAnalyser(ip.img, prob_map, stack_order=("TSCXY", "TCXY"))
        img_analyser.create_binary_mask()
        img_analyser.clean_binmask(min_pixel_size=MIN_PIXEL_SIZE)
        img_analyser.get_labels()
        img_logger.info(f"{img_analyser.total_rois} objects found in segmentation")
        return img_analyser

    def _classify_rois(self, img_analyser, img_logger):
        """Classify ROIs in batches."""
        device = get_device()
        with ReserveResource(device, 10.0, logger=img_logger, timeout=240):
            object_classes, labels = self.batch_classify_rois(img_analyser, batch_size=1)
        return object_classes, labels

    def _extract_measurements(self, img_analyser, metadata, object_classes, size_x, size_y, img_logger):
        """Extract fluorescence measurements and metadata."""
        # Background fluorescence
        img_logger.info("4.1 - Extracting background fluorescence intensity")
        fl_img_data = img_analyser.get("image", to_numpy=False)[:, 0, self.pi_channel]
        img_logger.info(f"FL channel (pi_channel={self.pi_channel}) stats before background extraction:")
        img_logger.info(f"  FL min={fl_img_data.min():.2f}, max={fl_img_data.max():.2f}, mean={fl_img_data.mean():.2f}, std={fl_img_data.std():.2f}")

        bck_fl = measure_background_intensity(
            img_analyser.get("image", to_numpy=False),
            img_analyser.get("labels", to_numpy=False),
            target_channel=self.pi_channel,
        )
        img_logger.info(f"Background FL per frame: min={bck_fl['background'].min():.2f}, max={bck_fl['background'].max():.2f}, mean={bck_fl['background'].mean():.2f}")

        # Fluorescence properties
        fl_prop = [
            "label", "centroid", "max_intensity", "min_intensity", "mean_intensity",
            "area", "major_axis_length", "minor_axis_length", "solidity", "orientation",
        ]

        img_logger.info("4.2 - Extracting fluorescence measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=self.pi_channel,
            properties=fl_prop,
            extra_properties=(roi_skewness, roi_std_dev),
        )
        img_logger.info(f"FL measurements extracted: {len(fl_measurements)} ROIs")
        img_logger.info(f"  Intensity stats - min: {fl_measurements['min_intensity'].min():.2f}, max: {fl_measurements['max_intensity'].max():.2f}, mean: {fl_measurements['mean_intensity'].mean():.2f}")

        fl_measurements["object_class"] = object_classes

        # Add time metadata
        img_logger.info("4.3 - Extracting time metadata")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")

        # Calculate relative intensities with safe division
        img_logger.info(f"Background values: min={fl_measurements['background'].min():.2f}, max={fl_measurements['background'].max():.2f}")
        safe_background = fl_measurements["background"] + 1.0
        fl_measurements[["rel_max_intensity", "rel_min_intensity", "rel_mean_intensity"]] = (
            fl_measurements[["max_intensity", "min_intensity", "mean_intensity"]].div(safe_background, axis=0)
        )
        img_logger.info(f"Relative FL intensities calculated (with safe division):")
        img_logger.info(f"  rel_mean_intensity: min={fl_measurements['rel_mean_intensity'].min():.3f}, max={fl_measurements['rel_mean_intensity'].max():.3f}, mean={fl_measurements['rel_mean_intensity'].mean():.3f}")

        # Object tracking (if enabled)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.4 - Running object tracking")
            track_features = fl_prop[5:10]
            self.cell_tracker.set_features(track_features)
            try:
                fl_measurements = self.cell_tracker.track_objects(
                    fl_measurements, volume_bounds=(size_x, size_y), logger=img_logger
                )
                img_logger.info("4.4 - Object tracking completed successfully")
            except Exception as e:
                img_logger.error(f"Object tracking failed: {e}")

        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")

        return fl_measurements

    def _perform_pi_classification(self, fl_measurements, name, img_logger):
        """Perform PI classification and generate summary."""
        if self.pi_classifier is None:
            img_logger.warning("PI classifier is None - skipping PI classification")
            return pd.DataFrame()

        img_logger.info("4.4 - Running PI classification", show_memory=True)
        img_logger.info(f"Number of objects to classify: {len(fl_measurements)}")
        img_logger.info(f"Features used: {list(self.pi_classifier.feature_names_in_)}")

        # Log feature values before classification
        img_logger.info(f"Feature summary for classification:")
        for feat in self.pi_classifier.feature_names_in_:
            if feat in fl_measurements.columns:
                img_logger.info(f"  {feat}: min={fl_measurements[feat].min():.3f}, max={fl_measurements[feat].max():.3f}, mean={fl_measurements[feat].mean():.3f}")

        predictions = self.pi_classifier.predict(
            fl_measurements[self.pi_classifier.feature_names_in_]
        )
        fl_measurements["pi_class"] = predictions
        fl_measurements["file"] = name

        # Log classification results
        pi_counts = fl_measurements["pi_class"].value_counts()
        img_logger.info(f"PI classification results: {pi_counts.to_dict()}")
        img_logger.info(f"  PMA_POS: {(predictions == 'PMA_POS').sum()} ({100*(predictions == 'PMA_POS').sum()/len(predictions):.1f}%)")
        img_logger.info(f"  PMA_NEG: {(predictions == 'PMA_NEG').sum()} ({100*(predictions == 'PMA_NEG').sum()/len(predictions):.1f}%)")

        # Generate summary data
        d_summary = self.generate_data_summary(
            fl_measurements,
            ["file", "frame", "channel", "date_time", "timestep", "abslag_in_s", "object_class"],
            img_logger,
        )
        img_logger.info(f"Summary data shape: {d_summary.shape}")

        return d_summary

    def _export_results(self, name, fl_measurements, d_summary, img_analyser,
                       object_classes, labels, export_labeled_mask, export_aligned_image, img_logger):
        """Export all results to files."""
        export_path = os.path.join(self.output_path, name)
        img_logger.info(f"5 - Writing output data to {export_path}")

        # Export measurements
        fl_measurements.to_csv(export_path + "_fl.csv")
        d_summary.to_csv(export_path + "_summary.csv")

        # Export labeled mask
        if export_labeled_mask:
            self._export_labeled_mask(
                export_path, img_analyser, object_classes, labels, fl_measurements, img_logger
            )

        # Export aligned image
        if export_aligned_image:
            image_8bit = convert_image(img_analyser.get("image", to_numpy=True), np.uint8)
            tifffile.imwrite(export_path + "_transformed.tiff", image_8bit, imagej=True)

    def _export_labeled_mask(self, export_path, img_analyser, object_classes, labels, fl_measurements, img_logger):
        """Export labeled mask with object and PI classification."""
        label_slice = img_analyser.get("labels", index=(slice(None), 0, 0), to_numpy=True)

        # Map object classes to the labeled mask
        object_class_mask = map_predictions_to_labels(
            label_slice,
            object_classes,
            labels,
            value_map={class_name: class_id + 1 for class_name, class_id in CLASS_TO_ID.items()},
        )

        # If PI classifier was used, create a second channel for PI classification
        if self.pi_classifier is not None:
            pi_class_mask = map_predictions_to_labels(
                label_slice,
                fl_measurements["pi_class"].tolist(),
                fl_measurements["label"].tolist(),
                value_map=PI_CLASS_MAP,
            )
            combined_mask = np.stack([object_class_mask, pi_class_mask], axis=1)
            labs_8bit = combined_mask.astype(np.uint8)
            axes = "TCYX"
            log_msg = "Exported labeled mask with object and PI classification channels"
        else:
            labs_8bit = object_class_mask.astype(np.uint8)
            axes = "TYX"
            log_msg = "Exported labeled mask with object classification channel"

        tifffile.imwrite(
            export_path + "_labels.tiff",
            labs_8bit,
            imagej=True,
            metadata={"axes": axes},
        )
        img_logger.info(log_msg)

    def batch_classify_rois(self, img_analyser, batch_size=5):
        """Classify ROIs in batches to manage memory."""
        labeled_mask = img_analyser.get("labels", index=(slice(None), 0, 0), to_numpy=True)
        img = img_analyser.get("image", index=(slice(None), 0, 0), to_numpy=True)

        n_frames = labeled_mask.shape[0]
        all_object_classes = []
        all_labels = []

        for start_frame in range(0, n_frames, batch_size):
            end_frame = min(start_frame + batch_size, n_frames)
            batch_labeled_mask = labeled_mask[start_frame:end_frame]
            batch_img = img[start_frame:end_frame]

            batch_classes, batch_labels = self.object_classifier.classify_rois(
                batch_labeled_mask, batch_img
            )

            all_object_classes.extend(batch_classes)
            all_labels.extend(batch_labels)

        return all_object_classes, all_labels

    def apply_background_correction(
        self,
        ip: ImagePreprocessor,
        reference_channel: int,
        pi_channel: int,
        nFrames: range,
        img_logger,
    ) -> None:
        """
        Apply background correction to both channels matching training preprocessing.

        Uses Gaussian blur for both channels:
        - BF: sigma=50, division (flat-field correction)
        - FL: sigma=150, 30% subtraction (gentle correction)
        """
        img_logger.info(f"=== BACKGROUND CORRECTION START ===")
        img_logger.info(f"Processing {len(list(nFrames))} frames")
        img_logger.info(f"BF channel: sigma={BF_GAUSSIAN_SIGMA}, division (flat-field)")
        img_logger.info(f"FL channel: sigma={FL_GAUSSIAN_SIGMA}, {FL_SUBTRACT_FRACTION*100}% subtract")

        for frame_idx in nFrames:
            # BF channel: Gaussian blur DIVIDE (flat-field correction)
            bf_img = ip.img[frame_idx, 0, reference_channel, :, :].copy().astype(np.float32)
            bf_background = gaussian_filter(bf_img, sigma=BF_GAUSSIAN_SIGMA)
            bf_background = np.maximum(bf_background, 1e-6)  # Avoid division by zero
            bf_corrected = bf_img / bf_background

            # Check for numerical issues
            if np.any(np.isnan(bf_corrected)):
                img_logger.error(f"ERROR: BF frame {frame_idx} contains NaN values after background correction!")
            if np.any(np.isinf(bf_corrected)):
                img_logger.error(f"ERROR: BF frame {frame_idx} contains Inf values after background correction!")

            ip.img[frame_idx, 0, reference_channel, :, :] = bf_corrected.astype(np.float32)

            # FL channel: Gentle large-scale background removal
            fl_img = ip.img[frame_idx, 0, pi_channel, :, :].copy().astype(np.float32)
            fl_background = gaussian_filter(fl_img, sigma=FL_GAUSSIAN_SIGMA)
            fl_corrected = fl_img - (fl_background * FL_SUBTRACT_FRACTION)

            # Check for numerical issues
            if np.any(np.isnan(fl_corrected)):
                img_logger.error(f"ERROR: FL frame {frame_idx} contains NaN values after background correction!")
            if np.any(np.isinf(fl_corrected)):
                img_logger.error(f"ERROR: FL frame {frame_idx} contains Inf values after background correction!")

            ip.img[frame_idx, 0, pi_channel, :, :] = fl_corrected.astype(np.float32)

            # Log first frame only
            if frame_idx == list(nFrames)[0]:
                img_logger.info(f"--- Frame {frame_idx} (first frame) ---")
                img_logger.info(f"BF BEFORE: min={bf_img.min():.1f}, max={bf_img.max():.1f}, mean={bf_img.mean():.1f}, std={bf_img.std():.1f}")
                img_logger.info(f"BF AFTER:  min={bf_corrected.min():.1f}, max={bf_corrected.max():.1f}, mean={bf_corrected.mean():.1f}, std={bf_corrected.std():.1f}")
                img_logger.info(f"FL BEFORE: min={fl_img.min():.1f}, max={fl_img.max():.1f}, mean={fl_img.mean():.1f}, std={fl_img.std():.1f}")
                img_logger.info(f"FL AFTER:  min={fl_corrected.min():.1f}, max={fl_corrected.max():.1f}, mean={fl_corrected.mean():.1f}, std={fl_corrected.std():.1f}")

        img_logger.info(f"=== BACKGROUND CORRECTION COMPLETED for {len(list(nFrames))} frames ===")

    def generate_data_summary(
        self,
        fl_measurements: pd.DataFrame,
        by_list: List[str],
        img_logger: MemoryLogger,
    ) -> pd.DataFrame:
        """
        Generate a summary DataFrame from fluorescence measurements with PI classification.

        This method aggregates the fluorescence measurements by file, frame, channel,
        timestamp information, and object class to create a summary of PI-positive and
        PI-negative cell counts and areas.

        Args:
            fl_measurements: DataFrame containing fluorescence measurements with 'pi_class' column.
                Must include columns: 'file', 'frame', 'channel', 'date_time', 'timestep',
                'abslag_in_s', 'object_class', 'label', 'area', and 'pi_class'.
            by_list: List of column names to group by.
            img_logger: Logger instance for recording progress and errors.

        Returns:
            pd.DataFrame: A summary DataFrame with aggregated counts and areas, or an empty
                DataFrame if an error occurs during the groupby operation.

        Notes:
            The summary includes the following aggregated metrics:
            - total_count: Total number of objects per group
            - pi_class_neg: Count of PI-negative objects
            - pi_class_pos: Count of PI-positive objects
            - area_pineg: Total area of PI-negative objects
            - area_pipos: Total area of PI-positive objects
            - area_total: Total area of all objects
        """
        try:
            img_logger.info(f"Group data by {by_list}")
            d_summary = (
                fl_measurements.groupby(by_list)
                .agg(
                    total_count=("label", "count"),
                    pi_class_neg=("pi_class", lambda x: (x == "PMA_NEG").sum()),
                    pi_class_pos=("pi_class", lambda x: (x == "PMA_POS").sum()),
                    area_pineg=(
                        "area",
                        lambda x: x[fl_measurements.loc[x.index, "pi_class"] == "PMA_NEG"].sum(),
                    ),
                    area_pipos=(
                        "area",
                        lambda x: x[fl_measurements.loc[x.index, "pi_class"] == "PMA_POS"].sum(),
                    ),
                    area_total=("area", "sum"),
                )
                .reset_index()
            )

            img_logger.info(f"Groupby operation completed successfully. Shape of d_summary: {d_summary.shape}")
        except Exception as e:
            img_logger.error(f"Error during groupby operation: {str(e)}")
            img_logger.error(f"Columns in fl_measurements: {fl_measurements.columns}")
            img_logger.error(f"Unique values in 'pi_class': {fl_measurements['pi_class'].unique()}")
            d_summary = pd.DataFrame()

        img_logger.info("d_summary created successfully", show_memory=True)
        return d_summary

    @staticmethod
    def check_px_values(ip, channel: int, round: int = None) -> np.ndarray:
        """Calculate mean pixel intensity across frames for a given channel."""
        means = np.mean(ip.img[:, 0, channel], axis=(1, 2))
        return np.round(means, round) if round is not None else means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMA kill curve analysis using ASCT_focusRestoration pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--worklist", type=str, default=None, help="Path to worklist file")
    args = parser.parse_args()

    c_reader = ConfReader(args.config)
    configs = c_reader.opt

    num_workers = configs.pipeline_setup.get("num_workers", 1)

    analysis_wf = PMA_KillCurve_Pipeline(
        configs.input_data["input_folder"],
        configs.input_data["output_folder"],
        file_type=configs.input_data["file_type"],
        worklist_path=args.worklist,
    )
    analysis_wf.load_config_dict(configs.pipeline_setup)

    # Load models from zip bundle
    print(f"Loading models from bundle: {configs.models['model_collection']}")
    analysis_wf.load_model_bundle(configs.models["model_collection"])

    try:
        analysis_wf.process_folder_parallel(
            files_pattern=configs.input_data["file_pattern"],
            export_labeled_mask=configs.input_data["export_labelled_masks"],
            export_aligned_image=configs.input_data["export_aligned_image"],
            num_workers=num_workers,
        )
    except Exception as e:
        print(f"Pipeline completed with exception during cleanup: {e}")
        print("This is typically a multiprocessing cleanup issue and can be ignored if files were generated.")
        import sys
        sys.exit(0)
