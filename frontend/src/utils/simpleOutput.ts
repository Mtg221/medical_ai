import type { DocumentDetailResponse, ExtractedEntity, ExtractedRelation } from "@/types/api";
import { getEntityText } from "@/utils/entityStyles";

const MERGEABLE_REL_TYPES = new Set([
  "HAS_VALUE",
  "HAS_UNIT",
  "HAS_DOSAGE",
  "HAS_FREQUENCY",
  "HAS_ROUTE",
]);

export function isEntityApproved(status: string | undefined): boolean {
  const st = String(status || "").toUpperCase();
  return st === "APPROVED" || st === "APPROVE_AS_GROUND_TRUTH" || st === "CORRECTED";
}

export function isEntityRejected(status: string | undefined): boolean {
  const st = String(status || "").toUpperCase();
  return st === "REJECTED" || st === "REJECT";
}

export function buildSimpleKeyValueLines(
  detail: DocumentDetailResponse | undefined,
  approvedOnly: boolean
): Array<{ key: string; value: string; approved: boolean }> {
  if (!detail) return [];
  const lines: Array<{ key: string; value: string; approved: boolean }> = [];
  if (detail.document.subject_id) {
    lines.push({ key: "patient", value: detail.document.subject_id, approved: true });
  }

  const kept = detail.entities.filter((e) => {
    if (isEntityRejected(e.review_status)) return false;
    if (approvedOnly) return isEntityApproved(e.review_status);
    return true;
  });

  const byId = new Map<string, ExtractedEntity>(kept.map((e) => [e.entity_id, e]));
  const childrenByHead = new Map<string, ExtractedEntity[]>();
  const consumedIds = new Set<string>();

  for (const rel of (detail.relations ?? []) as ExtractedRelation[]) {
    if (!MERGEABLE_REL_TYPES.has(rel.relation_type)) continue;
    const head = byId.get(rel.source_entity_id);
    const tail = byId.get(rel.target_entity_id);
    if (!head || !tail || head.entity_id === tail.entity_id) continue;
    const arr = childrenByHead.get(head.entity_id) ?? [];
    if (!arr.some((x) => x.entity_id === tail.entity_id)) {
      arr.push(tail);
      childrenByHead.set(head.entity_id, arr);
      consumedIds.add(tail.entity_id);
    }
  }

  const sortedKept = kept.slice().sort((a, b) => a.char_start - b.char_start);
  for (const ent of sortedKept) {
    if (consumedIds.has(ent.entity_id)) continue;
    const approved = isEntityApproved(ent.review_status);
    const text = getEntityText(ent);
    const kids = (childrenByHead.get(ent.entity_id) ?? []).slice().sort((a, b) => a.char_start - b.char_start);

    const tags: string[] = [];
    if (ent.certainty === "absent" || ent.assertion_status === "NEGATED") tags.push("absent");
    else if (ent.certainty === "possible" || ent.assertion_status === "UNCERTAIN") tags.push("possible");
    if (ent.temporality === "historical") tags.push("historical");
    if (ent.experiencer === "family" || ent.assertion_status === "FAMILY_HISTORY") tags.push("family");
    const tagSuffix = tags.length > 0 ? ` (${tags.join(", ")})` : "";

    if (ent.entity_type === "LAB_TEST" && kids.length > 0) {
      lines.push({
        key: `${text.toLowerCase()}${tagSuffix}`,
        value: kids.map((k) => getEntityText(k)).join(" "),
        approved,
      });
    } else if (ent.entity_type === "MEDICATION" && kids.length > 0) {
      lines.push({
        key: `medication${tagSuffix}`,
        value: [text, ...kids.map((k) => getEntityText(k))].join(" "),
        approved,
      });
    } else {
      const prettyType = ent.entity_type.toLowerCase().replace(/_/g, " ");
      lines.push({
        key: `${prettyType}${tagSuffix}`,
        value: text,
        approved,
      });
    }
  }
  return lines;
}

export interface ClinicalRecordJson {
  patient: {
    id: string;
  };
  encounter: {
    document: string;
    date: string | null;
    language: string;
  };
  symptoms: Array<Record<string, string>>;
  medical_history: Array<Record<string, string>>;
  diagnoses: Array<Record<string, string>>;
  medications: Array<Record<string, string>>;
  laboratory_and_vitals: Array<Record<string, string>>;
  examinations: Array<Record<string, string>>;
  uncertainties: string[];
}

export function buildClinicalRecordJson(
  detail: DocumentDetailResponse | undefined,
  approvedOnly: boolean
): ClinicalRecordJson | null {
  if (!detail) return null;

  const kept = detail.entities.filter((e) => {
    if (isEntityRejected(e.review_status)) return false;
    if (approvedOnly) return isEntityApproved(e.review_status);
    return true;
  });

  const byId = new Map<string, ExtractedEntity>(kept.map((e) => [e.entity_id, e]));
  const relsByHead = new Map<string, Array<{ relType: string; tail: ExtractedEntity }>>();
  const consumedIds = new Set<string>();

  for (const rel of (detail.relations ?? []) as ExtractedRelation[]) {
    const head = byId.get(rel.source_entity_id);
    const tail = byId.get(rel.target_entity_id);
    if (!head || !tail || head.entity_id === tail.entity_id) continue;
    const arr = relsByHead.get(head.entity_id) ?? [];
    arr.push({ relType: rel.relation_type, tail });
    relsByHead.set(head.entity_id, arr);
    if (MERGEABLE_REL_TYPES.has(rel.relation_type)) {
      consumedIds.add(tail.entity_id);
    }
  }

  const record: ClinicalRecordJson = {
    patient: {
      id: detail.document.subject_id || "unknown",
    },
    encounter: {
      document: detail.document.original_filename || detail.document.document_id,
      date: detail.document.created_at ? detail.document.created_at.slice(0, 10) : null,
      language: detail.document.detected_language || "en",
    },
    symptoms: [],
    medical_history: [],
    diagnoses: [],
    medications: [],
    laboratory_and_vitals: [],
    examinations: [],
    uncertainties: [],
  };

  const seenSymptom = new Map<string, Record<string, string>>();
  const seenDiag = new Map<string, Record<string, string>>();
  const seenHist = new Map<string, Record<string, string>>();
  const seenMed = new Map<string, Record<string, string>>();
  const seenLab = new Set<string>();
  const seenExam = new Set<string>();
  const seenUnc = new Set<string>();

  const addUnc = (msg: string) => {
    if (!seenUnc.has(msg)) {
      seenUnc.add(msg);
      record.uncertainties.push(msg);
    }
  };

  const mergeFinding = (
    bucket: Map<string, Record<string, string>>,
    targetArr: Array<Record<string, string>>,
    key: string,
    item: Record<string, string>,
    label: string
  ) => {
    const existing = bucket.get(key);
    if (!existing) {
      bucket.set(key, item);
      targetArr.push(item);
      return;
    }
    if (existing.assertion && item.assertion && existing.assertion !== item.assertion) {
      if (existing.assertion !== "conflicting") {
        addUnc(`Conflicting assertion for "${label}": ${existing.assertion} vs ${item.assertion}`);
        existing.assertion = "conflicting";
      }
    }
    if (!existing.anatomy && item.anatomy) existing.anatomy = item.anatomy;
    if (!existing.temporal_status && item.temporal_status) existing.temporal_status = item.temporal_status;
    if (!existing.experiencer && item.experiencer) existing.experiencer = item.experiencer;
  };

  const sortedKept = kept.slice().sort((a, b) => a.char_start - b.char_start);
  for (const ent of sortedKept) {
    const text = getEntityText(ent);
    const lowKey = text.trim().toLowerCase();
    const approved = isEntityApproved(ent.review_status);
    if (!approved && (ent.review_status === "needs_review" || ent.review_reason)) {
      const reason = ent.review_reason || "needs_review";
      addUnc(`${reason}: "${text}" (${ent.entity_type})`);
    }
    if (consumedIds.has(ent.entity_id)) continue;

    const kids = relsByHead.get(ent.entity_id) ?? [];
    const assertion =
      ent.certainty === "absent" || ent.assertion_status === "NEGATED"
        ? "absent"
        : ent.certainty === "possible" || ent.assertion_status === "UNCERTAIN"
        ? "possible"
        : "present";
    const isHistorical = ent.temporality === "historical" || ent.temporal_status === "HISTORICAL";
    const isFamily = ent.experiencer === "family" || ent.assertion_status === "FAMILY_HISTORY";

    if (ent.entity_type === "SYMPTOM") {
      const item: Record<string, string> = { entity: text, assertion };
      const anat = kids.find((k) => k.relType === "LOCATED_IN");
      if (anat) item.anatomy = getEntityText(anat.tail);
      if (isHistorical) item.temporal_status = "historical";
      if (isFamily) item.experiencer = "family";
      mergeFinding(seenSymptom, record.symptoms, lowKey, item, text);
    } else if (ent.entity_type === "DIAGNOSIS") {
      const item: Record<string, string> = { entity: text, assertion };
      if (isHistorical) item.temporal_status = "historical";
      if (isFamily) item.experiencer = "family";
      if (isHistorical || isFamily) {
        mergeFinding(seenHist, record.medical_history, `${lowKey}:${isFamily ? "fam" : "hist"}`, item, text);
      } else {
        mergeFinding(seenDiag, record.diagnoses, lowKey, item, text);
      }
    } else if (ent.entity_type === "MEDICATION") {
      const item: Record<string, string> = { name: text };
      const dose = kids.find((k) => k.relType === "HAS_DOSAGE");
      const freq = kids.find((k) => k.relType === "HAS_FREQUENCY");
      const route = kids.find((k) => k.relType === "HAS_ROUTE");
      if (dose) item.dose = getEntityText(dose.tail);
      if (freq) item.frequency = getEntityText(freq.tail);
      if (route) item.route = getEntityText(route.tail);
      if (isHistorical) item.status = "historical_or_discontinued";
      else if (assertion !== "present") item.status = assertion === "absent" ? "not taking" : assertion;
      const existingMed = seenMed.get(lowKey);
      if (!existingMed) {
        seenMed.set(lowKey, item);
        record.medications.push(item);
      } else {
        if (!existingMed.dose && item.dose) existingMed.dose = item.dose;
        if (!existingMed.frequency && item.frequency) existingMed.frequency = item.frequency;
        if (!existingMed.route && item.route) existingMed.route = item.route;
        if (existingMed.status !== item.status && (existingMed.status || item.status)) {
          existingMed.status = "conflicting";
          addUnc(`Conflicting medication status for "${text}"`);
        }
      }
    } else if (ent.entity_type === "LAB_TEST") {
      const item: Record<string, string> = { test: text };
      const val = kids.find((k) => k.relType === "HAS_VALUE");
      const unit = kids.find((k) => k.relType === "HAS_UNIT");
      if (val) item.value = getEntityText(val.tail);
      if (unit) item.unit = getEntityText(unit.tail);
      if (assertion !== "present") item.assertion = assertion;
      const sig = `${lowKey}|${item.value ?? ""}|${item.unit ?? ""}`;
      if (!seenLab.has(sig)) {
        seenLab.add(sig);
        record.laboratory_and_vitals.push(item);
      }
    } else if (ent.entity_type === "PROCEDURE") {
      const item: Record<string, string> = { name: text };
      const reveals = kids.find((k) => k.relType === "REVEALS");
      if (reveals) item.finding = getEntityText(reveals.tail);
      if (assertion !== "present") item.status = assertion === "possible" ? "pending_or_uncertain" : assertion;
      const sig = `${lowKey}|${item.finding ?? ""}|${item.status ?? ""}`;
      if (!seenExam.has(sig)) {
        seenExam.add(sig);
        record.examinations.push(item);
      }
    } else if (ent.entity_type === "DATE" && !record.encounter.date && ent.normalized_text) {
      record.encounter.date = ent.normalized_text;
    }
  }

  return record;
}
