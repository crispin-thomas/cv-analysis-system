import type React from "react";
import { useState } from "react";
import { uploadCV } from "../api";

export function CVUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    try {
      const result = await uploadCV(file);
      setUploadStatus({
        success: true,
        message: `CV uploaded successfully with ID: ${result.filename}`,
      });
      setFile(null);
      // Reset the file input
      const fileInput = document.getElementById("cv-file") as HTMLInputElement;
      if (fileInput) fileInput.value = "";
      window.location.reload()
    } catch (error) {
      setUploadStatus({
        success: false,
        message: error instanceof Error ? error.message : "Failed to upload CV",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="cv-upload-container">
      <h2 className="cv-upload-title">Upload CV</h2>
      <p className="cv-upload-description">
        Upload a CV file to add it to the system. Supported formats: PDF, DOCX.
      </p>

      {uploadStatus && (
        <div
          className={`cv-upload-status ${
            uploadStatus.success ? "success" : "error"
          }`}
        >
          <strong>{uploadStatus.success ? "Success" : "Error"}</strong>
          <p>{uploadStatus.message}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="cv-upload-form">
        <div className="cv-upload-file-input">
          <label htmlFor="cv-file" className="cv-upload-label">
            CV File
          </label>
          <input
            id="cv-file"
            type="file"
            accept=".pdf,.docx,.doc"
            onChange={handleFileChange}
            className="cv-upload-input"
            required
          />
        </div>

        <div className="cv-upload-button-container">
          <button
            type="submit"
            disabled={!file || uploading}
            className="cv-upload-button"
          >
            {uploading ? "Uploading..." : "Upload CV"}
          </button>
        </div>
      </form>

      {file && (
        <div className="cv-upload-file-info">
          <span>Selected: {file.name}</span>
        </div>
      )}
    </div>
  );
}
