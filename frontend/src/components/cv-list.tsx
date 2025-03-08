import { Eye, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Button, Modal, Table } from "rsuite";
import { deleteCV, fetchCVById, fetchCVs } from "../api";
import type { _CV } from "../types";
import ViewCVModal from "./view-cv-modal";

type Props = {
  height: number;
};

export function CVList({ height }: Props) {
  const [cvs, setCVs] = useState<_CV[]>([]); // Array of CVs
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCV, setSelectedCV] = useState<_CV | null>(null);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [cvToDelete, setCVToDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const loadCVs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchCVs();
      setCVs(data.data); // Assuming new data format
    } catch (err) {
      setError("Failed to load CVs. Please try again later.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCVs();
  }, []);

  const handleViewCV = async (id: string) => {
    if (!id) return;
    try {
      const cv = await fetchCVById(id);
      setSelectedCV(cv);
      setViewDialogOpen(true);
    } catch (err) {
      console.error("Failed to fetch CV details:", err);
    }
  };

  const handleDeleteClick = (id: string) => {
    setCVToDelete(id);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!cvToDelete) return;
    setDeleting(true);
    try {
      await deleteCV(cvToDelete);
      setCVs(cvs.filter((cv) => cv.id !== cvToDelete)); // Remove the deleted CV
      setDeleteDialogOpen(false);
      setCVToDelete(null);
    } catch (err) {
      console.error("Failed to delete CV:", err);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div
      className={"cv-list-container"}
      style={{ height: height - 234 + "px" }}
    >
      <header className="cv-list-header">
        <h2>Manage CVs</h2>
        <p>Browse and manage uploaded CVs efficiently.</p>
      </header>

      {loading ? (
        <div className="loading-spinner">Loading...</div>
      ) : error ? (
        <div className="error-message">{error}</div>
      ) : cvs.length === 0 ? (
        <div className="no-cvs-message">No CVs found. Upload some CVs!</div>
      ) : (
        <Table
          height={height - 330}
          data={cvs}
          rowKey="id"
          bordered
          rowHeight={60}
          className="cv-table"
        >
          <Table.Column width={250} align="left">
            <Table.HeaderCell>Name</Table.HeaderCell>
            <Table.Cell>
              {(rowData) => rowData.personal_info.name || `CV-${rowData.id}`}
            </Table.Cell>
          </Table.Column>

          <Table.Column width={150} align="left">
            <Table.HeaderCell>File Name</Table.HeaderCell>
            <Table.Cell>{(rowData) => rowData.file_info.filename}</Table.Cell>
          </Table.Column>

          <Table.Column width={150} align="left">
            <Table.HeaderCell>File Size</Table.HeaderCell>
            <Table.Cell>
              {(rowData) => {
                const kb = rowData.file_info.file_size / 1024;
                if (kb < 1024) {
                  return `${kb.toFixed(2)} KB`;
                }
                const mb = kb / 1024;
                return `${mb.toFixed(2)} MB`;
              }}
            </Table.Cell>
          </Table.Column>

          <Table.Column width={150} align="center">
            <Table.HeaderCell>Uploaded On</Table.HeaderCell>
            <Table.Cell>
              {(rowData) =>
                new Date(rowData.file_info.upload_date).toLocaleDateString()
              }
            </Table.Cell>
          </Table.Column>

          <Table.Column width={200} align="center">
            <Table.HeaderCell>Actions</Table.HeaderCell>
            <Table.Cell>
              {(rowData) => (
                <div className="action-buttons">
                  <Button
                    onClick={() => handleViewCV(rowData.id)}
                    size="sm"
                    appearance="primary"
                    color="green"
                  >
                    <Eye />
                  </Button>
                  <Button
                    className="delete-button"
                    onClick={() => handleDeleteClick(rowData.id)}
                    size="sm"
                    color="red"
                    appearance="primary"
                  >
                    <Trash2 />
                  </Button>
                </div>
              )}
            </Table.Cell>
          </Table.Column>
        </Table>
      )}

      <ViewCVModal
        onClose={() => setViewDialogOpen(false)}
        open={viewDialogOpen}
        selectedCV={selectedCV}
      />

      {/* Delete CV Modal */}
      <Modal open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <Modal.Header>
          <h3>Are you sure?</h3>
        </Modal.Header>
        <Modal.Body>
          <p>This action will permanently delete the CV.</p>
        </Modal.Body>
        <Modal.Footer>
          <Button
            onClick={() => setDeleteDialogOpen(false)}
            disabled={deleting}
            appearance="primary"
            color="green"
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            disabled={deleting}
            color="red"
            appearance="primary"
          >
            {deleting ? "Deleting..." : "Delete"}
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}
