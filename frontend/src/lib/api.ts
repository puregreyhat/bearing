export type Task = "binary" | "multiclass"
export type Model = "rf" | "svm" | "xgboost"

export type DiagnosisResult = {
  test_id: string
  task: Task
  model_family: Model
  classes: string[]
  input_filename: string | null
  source_sampling_rate: number
  processed_sampling_rate: number
  sample_count: number
  duration_seconds: number
  window_size: number
  valid_window_count: number
  discarded_tail_samples: number
  probabilities: Record<string, number>
  window_probabilities: Record<string, number>[]
  window_predictions: string[]
  predicted_class: string
  confidence: number
  health_index: number | null
  first_window: number[]
  first_window_features: Record<string, number>
  warnings: string[]
}

export type AdminTestLog = {
  test_id: string
  created_at: string
  input_filename: string
  model_family: Model
  predicted_result: string
}

export type AdminTestLogResponse = {
  items: AdminTestLog[]
  total: number
  limit: number
  offset: number
}

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

export async function diagnose(input: {
  task: Task
  model: Model
  samplingRate: number
  sampleName?: string
  file?: File
}): Promise<DiagnosisResult> {
  const body = new FormData()
  body.append("task", input.task)
  body.append("model_family", input.model)
  body.append("sampling_rate", String(input.samplingRate))
  if (input.file) body.append("signal_file", input.file)
  else body.append("sample_name", input.sampleName || "097_normal.csv")

  const response = await fetch(`${API_BASE}/api/diagnose`, { method: "POST", body })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || "Diagnosis request failed")
  return payload as DiagnosisResult
}

export async function getAdminTests(search = ""): Promise<AdminTestLogResponse> {
  const query = new URLSearchParams({ limit: "100", offset: "0" })
  if (search.trim()) query.set("search", search.trim())
  const response = await fetch(`${API_BASE}/api/admin/tests?${query}`)
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || "Could not load test logs")
  return payload as AdminTestLogResponse
}
