//! Host-visible video ready/audio-track comparison helpers.

use renderide_shared::{VideoAudioTrack, VideoTextureReady};

/// Compares a `VideoTextureReady` message to another.
pub(super) fn video_texture_ready_eq(a: &VideoTextureReady, b: &VideoTextureReady) -> bool {
    a.asset_id == b.asset_id
        && a.has_alpha == b.has_alpha
        && a.instance_changed == b.instance_changed
        && a.size == b.size
        && a.length.to_bits() == b.length.to_bits()
        && a.playback_engine == b.playback_engine
        && video_audio_tracks_eq(&a.audio_tracks, &b.audio_tracks)
}

/// Compares audio track slices.
pub(super) fn video_audio_tracks_eq(a: &[VideoAudioTrack], b: &[VideoAudioTrack]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a_track, b_track)| video_audio_track_eq(a_track, b_track))
}

/// Compares a `VideoAudioTrack` to another.
pub(super) fn video_audio_track_eq(a: &VideoAudioTrack, b: &VideoAudioTrack) -> bool {
    a.sample_rate == b.sample_rate
        && a.index == b.index
        && a.name == b.name
        && a.language_code == b.language_code
        && a.channel_count == b.channel_count
}
