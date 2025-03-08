import { QueryRequest, QueryResponse } from "../types";

const API_URL = "http://localhost:8000"

export async function queryCV(
  queryRequest: QueryRequest
): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/query/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(queryRequest),
  });

  if (!response.ok) {
    throw new Error(`Failed to query CVs: ${response.statusText}`);
  }

  return response.json();
}
