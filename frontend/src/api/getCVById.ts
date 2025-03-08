import type { _CV } from "../types"

const API_URL = "http://localhost:8000"

export async function fetchCVById(id: string): Promise<_CV> {
    const response = await fetch(`${API_URL}/cvs/${id}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
  
    if (!response.ok) {
      throw new Error(`Failed to fetch CV: ${response.statusText}`)
    }
  
    return response.json()
  }