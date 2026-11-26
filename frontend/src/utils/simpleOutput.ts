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
