// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2025-2026 ViralMint Contributors
import { useCallback, useEffect, useState } from "react"
import { storageKey } from "../../utils/storage"

/* ── The per-clip settings both Clipper surfaces share ─────────
   Captions, emoji density, silence trimming, vertical reframing and
   transcription apply per CLIP, identically whether the range came off the
   bench's timeline or the Auto-cut dialog's AI picker. They were
   duplicated across the two — same four controls, two states — so choosing
   a caption style in one place did nothing in the other, and whichever
   surface you happened to finish in decided the render.

   One hook, owned once by ClipStudio and handed to both, so the surfaces
   cannot disagree. Persisted per session because these are habits, not
   per-video decisions: a creator who wants Bold captions and vertical
   reframing wants them for the next video too.

   The VALUES are the API contract — `caption_style` ids come from
   CAPTION_STYLES (tests/test_caption_styles_parity.py pins them against
   the backend) and "none" means captions off, intercepted upstream of the
   ASS builder. Don't invent a new sentinel here.
*/

export const CLIP_SETTINGS_DEFAULTS = {
  // Most extractions are raw cuts the creator styles elsewhere, and
  // captions-off also skips the burn re-encode entirely — the fast path is
  // the default path.
  caption_style: "none",
  emoji_style: "moderate",
  remove_silence: false,
  force_vertical: false,
  vertical_fill: "zoom",
  whisper_quality: "balanced",
  force_retranscribe: false,
}

export default function useClipSettings() {
  const key = storageKey("clipper", "settings")
  const [settings, setSettings] = useState(() => {
    try {
      const raw = sessionStorage.getItem(key)
      const saved = raw ? JSON.parse(raw) : null
      // Merge over defaults, never replace: a stored blob from an older
      // build is missing whatever was added since.
      return saved && typeof saved === "object"
        ? { ...CLIP_SETTINGS_DEFAULTS, ...saved }
        : CLIP_SETTINGS_DEFAULTS
    } catch {
      return CLIP_SETTINGS_DEFAULTS
    }
  })

  useEffect(() => {
    try { sessionStorage.setItem(key, JSON.stringify(settings)) } catch { /* private mode */ }
  }, [key, settings])

  /** Patch one or more fields. */
  const update = useCallback((patch) => {
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  /** @see clipSettingsPayload — bound to this hook's state. */
  const toPayload = useCallback(
    (opts) => clipSettingsPayload(settings, opts),
    [settings],
  )

  return { settings, update, toPayload }
}

/**
 * The subset every extract request carries, in the shape
 * `ClipStudio.handleExtract` expects.
 *
 * `hasTranscript` is not decoration. handleExtract only forwards
 * `whisper_quality` when `force_retranscribe` is set — so on a source with no
 * cached transcript, where Whisper MUST run and the chosen model is the only
 * thing that decides its quality, a raw `{...settings}` sent nothing and the
 * backend fell back to "balanced". The bench's footer said "Whisper: best" and
 * meant "balanced". ExtractDialog had already grown its own private fix for
 * this; the bench had not, which is exactly the two-copies-of-one-decision
 * failure the shared settings object exists to prevent. Derive it once, here,
 * and let both surfaces call this.
 *
 * @param {object} settings  a CLIP_SETTINGS_DEFAULTS-shaped object
 * @param {{hasTranscript?: boolean}} [opts]  whether the source already has a
 *   cached Whisper transcript. Omitted → treated as absent, i.e. transcription
 *   is required, which is the safe direction: it sends the user's choice.
 */
export function clipSettingsPayload(settings, { hasTranscript = false } = {}) {
  return {
    caption_style: settings.caption_style,
    emoji_style: settings.emoji_style,
    remove_silence: settings.remove_silence,
    force_vertical: settings.force_vertical,
    vertical_fill: settings.vertical_fill,
    whisper_quality: settings.whisper_quality,
    force_retranscribe: !hasTranscript || !!settings.force_retranscribe,
  }
}
