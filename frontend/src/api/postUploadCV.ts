import { UploadCV } from "../types"

const API_URL = "http://localhost:8000"

export async function uploadCV(file: File): Promise<UploadCV> {
    const formData = new FormData()
    formData.append("cv_file", file)
  
    const response = await fetch(`${API_URL}/upload/`, {
      method: "POST",
      body: formData,
    })
  
    if (!response.ok) {
      throw new Error(`Failed to upload CV: ${response.statusText}`)
    }
  
    return response.json()
  }