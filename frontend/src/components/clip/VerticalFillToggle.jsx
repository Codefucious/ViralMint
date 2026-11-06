// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2025-2026 ViralMint Contributors
import { ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material"

const OPTIONS = [
  {
    value: "zoom",
    label: "Zoom to speaker",
    hint: "Crops in and follows the speaker. Anything at the sides, like on-screen text, is cut off.",
  },
  {
    value: "blur",
    label: "Blur background",
    hint: "Keeps the whole picture, with a blurred copy behind it. Slower to render.",
  },
]

/** How a landscape source fills a vertical frame. Shared by both clip surfaces. */
export default function VerticalFillToggle({ value, onChange }) {
  return (
    <ToggleButtonGroup
      exclusive
      size="small"
      color="primary"
      aria-label="Vertical fill"
      value={value}
      onChange={(_, next) => { if (next) onChange(next) }}
      sx={{ "& .MuiToggleButton-root": { py: 0.25, px: 1, textTransform: "none", fontSize: 12 } }}
    >
      {OPTIONS.map((o) => (
        <Tooltip key={o.value} describeChild title={o.hint}>
          <ToggleButton value={o.value}>{o.label}</ToggleButton>
        </Tooltip>
      ))}
    </ToggleButtonGroup>
  )
}
