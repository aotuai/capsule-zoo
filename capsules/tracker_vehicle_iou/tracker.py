from typing import List, Tuple
import logging

from .track import Track

from vcap import DetectionNode
from vcap_utils import iou_cost_matrix, linear_assignment

MATCH_FUNC_OUTPUT = Tuple[
    List[Tuple[DetectionNode, Track]],
    List[Track],
    List[DetectionNode]]


class Tracker:
    def __init__(self, min_iou, max_misses, n_hits_to_init):
        self.tracks: List[Track] = []
        self.min_iou_for_iou_match = min_iou
        """Minimum IOU in order for an 'iou only' match to occur"""
        self.max_misses = max_misses
        """How many frames in a row a detection can go before being 
        deleted"""

        self.n_hits_to_init = n_hits_to_init

    def predict(self):
        """Propogate track state distributions 1 time step forward."""

    def update(self, detections: List[DetectionNode]):
        matches, unmatched_tracks, unmatched_dets = self._match(
            new_dets=detections,
            tracks=self.tracks)

        for det, track in matches:
            track.update(det)

        for track in unmatched_tracks:
            track.mark_missed()

        for det in unmatched_dets:
            self.start_track(det)

        # Clean out dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted]

    def start_track(self, det: DetectionNode):
        track = Track(det, self.max_misses, self.n_hits_to_init)
        self.tracks.append(track)
        return track

    def _match(self, new_dets: List[DetectionNode], tracks: List[Track]) \
            -> MATCH_FUNC_OUTPUT:
        matches: List[Tuple[DetectionNode, Track]] = []

        confirmed_tracks = [t for t in tracks if t.is_confirmed]
        unconfirmed_tracks = [t for t in tracks if not t.is_confirmed]

        # Try to match dets with confirmed tracks
        matched_c, unmatched_c_tracks, unmatched_c_dets = \
            self._try_iou_matching(new_dets, confirmed_tracks)

        # Try to match dets with unconfirmed tracks
        matched_u, unmatched_u_tracks, unmatched_u_dets = \
            self._try_iou_matching(unmatched_c_dets, unconfirmed_tracks)

        return (matched_c + matched_u,
                unmatched_u_tracks + unmatched_c_tracks,
                unmatched_u_dets)

    def _try_iou_matching(self, dets: List[DetectionNode],
                          tracks: List[Track]) -> MATCH_FUNC_OUTPUT:
        # Match confirmed tracks with detections where possible, IOU based
        tracks_latest_dets = [t.latest_det for t in tracks]
        if len(tracks_latest_dets) == 0:
            return [], tracks, dets

        iou_cost = iou_cost_matrix(dets, tracks_latest_dets)

        iou_cost[iou_cost > (1 - self.min_iou_for_iou_match)] = 1
        indices = linear_assignment(iou_cost)

        matches: List[Tuple[DetectionNode, Track]] = []
        unmatched_tracks = tracks.copy()
        unmatched_dets = dets.copy()
        for det_index, conf_track_index in indices:
            det = dets[det_index]
            track = tracks[conf_track_index]

            cost_iou = iou_cost[det_index][conf_track_index]
            if cost_iou == 1:
                continue

            matches.append((det, track))
            unmatched_tracks.remove(track)
            unmatched_dets.remove(det)
        return matches, unmatched_tracks, unmatched_dets
