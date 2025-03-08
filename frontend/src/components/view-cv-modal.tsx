import { Modal, Button } from "rsuite";
import { _CV } from "../types";

type Props = {
  open: boolean;
  onClose: () => void;
  selectedCV: _CV | null;
};

const ViewCVModal = ({ open, onClose, selectedCV }: Props) => {
  if (!selectedCV) return null;

  return (
    <Modal open={open} onClose={onClose}>
      <Modal.Header>
        <h3>{selectedCV.personal_info.name}</h3>
      </Modal.Header>
      <Modal.Body className="view-cv-body">
        <p>
          <strong>Email:</strong> {selectedCV.personal_info.email}
        </p>
        <p>
          <strong>Phone:</strong> {selectedCV.personal_info.phone}
        </p>
        <p>
          <strong>Location:</strong> {selectedCV.personal_info.location}
        </p>

        <h4>Education</h4>
        <ul>
          {selectedCV.education.map((edu, idx) => (
            <li key={idx}>
              {edu.degree} in {edu.field} from {edu.institution} ({edu.dates})
            </li>
          ))}
        </ul>
        <h4>Work Experience</h4>
        <ul>
          {selectedCV.work_experience.map((work, idx) => (
            <li key={idx}>
              <strong>{work.position}</strong> at {work.company} ({work.dates})
              <ul>
                {work.responsibilities.map((resp, rIdx) => (
                  <li key={rIdx}>{resp}</li>
                ))}
              </ul>
            </li>
          ))}
        </ul>

        <h4>Skills</h4>
        <ul>
          {selectedCV.skills.map((skill, idx) => (
            <li key={idx}>
              {skill.name}
              {skill.proficiency.length > 0 ? "-" + skill.proficiency : ""}
            </li>
          ))}
        </ul>

        <h4>Projects</h4>
        <ul>
          {selectedCV.projects.map((project, idx) => (
            <li key={idx}>
              <strong>{project.name}</strong>
              {project.dates && (
                <>
                  <br />
                  <em>{project.dates}</em>
                </>
              )}
              {project.description && (
                <>
                  <br />
                  <p>{project.description}</p>
                </>
              )}
              {project.technologies?.length > 0 && (
                <>
                  <br />
                  <strong>Technologies:</strong>{" "}
                  {project.technologies.join(", ")}
                </>
              )}
            </li>
          ))}
        </ul>

        <h4>Certifications</h4>
        <ul>
          {selectedCV.certifications.map((certification, idx) => (
            <li key={idx}>
              <strong>{certification.name}</strong>
              {certification.date && (
                <>
                  <br />
                  <em>{certification.date}</em>
                </>
              )}
              {certification.issuer && (
                <>
                  <br />
                  Issued by: {certification.issuer}
                </>
              )}
            </li>
          ))}
        </ul>
      </Modal.Body>
      <Modal.Footer
        style={{
          borderTop: "1px solid #dfe0df",
          marginTop: "10px",
          paddingTop: "10px",
          marginBottom: -5,
        }}
      >
        <Button onClick={onClose} appearance="primary">
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default ViewCVModal;
