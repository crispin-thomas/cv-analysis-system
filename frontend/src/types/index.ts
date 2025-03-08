export interface Message {
  id: string;
  role: "user" | "system";
  content: string;
}

export interface QueryRequest {
  query: string;
  conversation_id: string;
  user_id: string;
}

export interface QueryResponse {
  response: string;
}

export interface UploadCV {
  cv_id: string;
  filename: string;
  status: string;
}

export interface CV {
  meta: Meta;
  data: _CV[];
}

export interface _CV {
  id: string;
  file_info: Fileinfo;
  personal_info: Personalinfo;
  education: Education[];
  work_experience: Workexperience[];
  skills: Skill[];
  projects: Project[];
  certifications: Certification[];
}

interface Certification {
  name: string;
  issuer: string;
  date: string;
}

interface Project {
  name: string;
  description: string;
  technologies: string[];
  dates: string;
}


interface Skill {
  name: string;
  proficiency: string;
}

interface Workexperience {
  company: string;
  position: string;
  dates: string;
  location?: string;
  responsibilities: string[];
}

interface Education {
  degree: string;
  field: string;
  institution: string;
  dates: string;
  location?: string;
}

interface Personalinfo {
  name: string;
  email: string;
  phone: string;
  location: string;
}

interface Fileinfo {
  filename: string;
  file_size: number;
  file_type: string;
  upload_date: string;
}

interface Meta {
  total_count: number;
}
