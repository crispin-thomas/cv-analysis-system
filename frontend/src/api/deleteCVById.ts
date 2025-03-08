const API_URL = "http://localhost:8000"

export async function deleteCV(id: string): Promise<void> {
    const response = await fetch(`${API_URL}/cvs/${id}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    })
  
    if (!response.ok) {
      throw new Error(`Failed to delete CV: ${response.statusText}`)
    }
  }
  