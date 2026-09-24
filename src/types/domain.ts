export type SampleSource = 'upload' | 'camera';

export interface DatasetClass {
  id: string;
  name: string;
  color: string;
  createdAt: number;
}

export type QualityWarning = 'low-resolution' | 'blurry' | 'near-duplicate';

export interface QualityReport {
  meanLuma: number;
  stdLuma: number;
  sharpness: number;
  warnings: QualityWarning[];
}

export interface ImageSample {
  id: string;
  projectId: string;
  classId: string;
  source: SampleSource;
  /** Original file name for an upload; a generated label like "camera-frame-003.jpg" for a capture. */
  name: string;
  width: number;
  height: number;
  mimeType: string;
  size: number;
  createdAt: number;
  /** Camera bursts share a groupId; every upload gets its own. */
  groupId: string;
  /** Order inside the group (camera frame index). */
  seq: number;
  sha256: string;
  phash: string;
  quality: QualityReport;
}

export interface Project {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  classes: DatasetClass[];
}

export type RejectReason =
  | 'corrupted'
  | 'unsupported-type'
  | 'too-large'
  | 'blank'
  | 'too-dark'
  | 'too-bright'
  | 'too-small'
  | 'duplicate'
  | 'label-conflict';

export interface IngestOutcome {
  name: string;
  accepted: boolean;
  reason?: RejectReason;
  detail?: string;
  warnings: QualityWarning[];
  sampleId?: string;
}

export type IssueSeverity = 'error' | 'warning' | 'info';
export interface DatasetIssue {
  severity: IssueSeverity;
  code: string;
  message: string;
  classId?: string;
}
