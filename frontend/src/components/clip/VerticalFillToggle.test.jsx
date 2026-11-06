import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import VerticalFillToggle from "./VerticalFillToggle"
import { CLIP_SETTINGS_DEFAULTS, clipSettingsPayload } from "./useClipSettings"

describe("VerticalFillToggle", () => {
  it("shows the current choice", () => {
    render(<VerticalFillToggle value="zoom" onChange={() => {}} />)
    expect(screen.getByRole("button", { name: "Zoom to speaker" }))
      .toHaveAttribute("aria-pressed", "true")
    expect(screen.getByRole("button", { name: "Blur background" }))
      .toHaveAttribute("aria-pressed", "false")
  })

  it("switches to the other option", () => {
    const onChange = vi.fn()
    render(<VerticalFillToggle value="zoom" onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Blur background" }))
    expect(onChange).toHaveBeenCalledWith("blur")
  })

  it("can't be left with nothing selected", () => {
    const onChange = vi.fn()
    render(<VerticalFillToggle value="zoom" onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Zoom to speaker" }))
    expect(onChange).not.toHaveBeenCalled()
  })
})

describe("clip settings", () => {
  it("default to zoom and send the choice with the request", () => {
    expect(CLIP_SETTINGS_DEFAULTS.vertical_fill).toBe("zoom")
    expect(clipSettingsPayload({ ...CLIP_SETTINGS_DEFAULTS, vertical_fill: "blur" }).vertical_fill)
      .toBe("blur")
  })
})
