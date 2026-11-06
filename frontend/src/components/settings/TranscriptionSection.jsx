import { useEffect, useState } from "react"
import {
  Alert, FormControl, InputLabel, MenuItem, Select, Stack, Typography,
} from "@mui/material"
import http from "../../api/http"
import useAppStore from "../../store/appStore"

const LOCAL = ""

export default function TranscriptionSection({ settings, updateSettings }) {
  const showSnackbar = useAppStore((s) => s.showSnackbar)
  const [models, setModels] = useState([])
  const [loadError, setLoadError] = useState("")
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    http.get("/api/config/transcription_models")
      .then(({ data }) => setModels(data?.models || []))
      .catch((e) => setLoadError(
        e?.response?.data?.detail || "Couldn't load OpenRouter's transcription models.",
      ))
  }, [])

  const selected = settings?.transcription_model || LOCAL
  const external = selected !== LOCAL
  // A saved model can leave OpenRouter's list, or the list can fail to load.
  // It still has to show as the current choice.
  const options = external && !models.some((m) => m.id === selected)
    ? [{ id: selected, name: selected }, ...models]
    : models

  const handleChange = async (e) => {
    const value = e.target.value
    setSaving(true)
    try {
      await updateSettings({ transcription_model: value })
      showSnackbar(value ? "Transcription model saved" : "Using local Whisper", "success")
    } catch (err) {
      // updateSettings already shows a snackbar on error
    } finally {
      setSaving(false)
    }
  }

  return (
    <Stack spacing={2}>
      <FormControl size="small" fullWidth>
        <InputLabel id="transcription-model-label" shrink>Transcription model</InputLabel>
        <Select
          labelId="transcription-model-label"
          label="Transcription model"
          notched
          displayEmpty
          value={selected}
          onChange={handleChange}
          disabled={saving}
        >
          <MenuItem value={LOCAL}>Local Whisper (free, runs on this machine)</MenuItem>
          {options.map((m) => (
            <MenuItem key={m.id} value={m.id}>{m.name}</MenuItem>
          ))}
        </Select>
      </FormControl>

      {loadError && <Alert severity="warning" variant="outlined">{loadError}</Alert>}

      {external && !settings?.openrouter_key_available && (
        <Alert severity="warning" variant="outlined">
          No OpenRouter key found, so local Whisper will be used. Add one under AI Provider
          or set <code>OPENROUTER_API_KEY</code> in your <code>.env</code>.
        </Alert>
      )}

      <Typography variant="caption" color="text.secondary">
        {external
          ? "Your audio is uploaded to OpenRouter and billed to your key. Captions need word " +
            "timings: if this model can't provide them, or the request fails, local Whisper " +
            "takes over and you'll see a warning in chat. Narration ViralMint generates " +
            "itself is always timed locally."
          : "Audio stays on this machine. Pick an OpenRouter model to transcribe in the cloud " +
            "instead, charged to your OpenRouter key."}
      </Typography>
    </Stack>
  )
}
