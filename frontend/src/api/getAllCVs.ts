import { CV } from "../types";

const API_URL = "http://localhost:8000"

export async function fetchCVs(): Promise<CV> {
  const response = await fetch(`${API_URL}/cvs/`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch CVs: ${response.statusText}`);
  }


  return response.json();
}
