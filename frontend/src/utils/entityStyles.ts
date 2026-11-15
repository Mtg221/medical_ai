import type { ExtractedEntity } from "@/types/api";

export interface EntityTypeStyle {
  type: string;
  label: string;
  shortCode: string;
  hex: string;
  highlightClass: string;
  chipClass: string;
}

export type CanonicalAssertion =
  | "present"
  | "absent"
  | "possible"
  | "hypothetical"
  | "family"
  | "historical";

export interface AssertionStyle {
  canonical: CanonicalAssertion;
  badge: string | null;
  showClock: boolean;
  textClass: string;
  borderStyleClass: string;
  ariaLabel: string;
}

export const ENTITY_TYPE_STYLES: Record<string, EntityTypeStyle> = {
  DIAGNOSIS: {
    type: "DIAGNOSIS",
    label: "Diagnosis",
    shortCode: "DX",
    hex: "#dc2626",
    highlightClass: "bg-rose-100 text-rose-950 border-rose-500 dark:bg-rose-950/60 dark:text-rose-100 dark:border-rose-400",
    chipClass: "bg-rose-100 text-rose-800 border-rose-300 dark:bg-rose-950 dark:text-rose-200 dark:border-rose-700",
  },
  SYMPTOM: {
    type: "SYMPTOM",
    label: "Symptom",
    shortCode: "SX",
    hex: "#d97706",
    highlightClass: "bg-amber-100 text-amber-950 border-amber-500 dark:bg-amber-950/60 dark:text-amber-100 dark:border-amber-400",
    chipClass: "bg-amber-100 text-amber-900 border-amber-300 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-700",
  },
  MEDICATION: {
    type: "MEDICATION",
    label: "Medication",
    shortCode: "RX",
    hex: "#0284c7",
    highlightClass: "bg-sky-100 text-sky-950 border-sky-500 dark:bg-sky-950/60 dark:text-sky-100 dark:border-sky-400",
    chipClass: "bg-sky-100 text-sky-900 border-sky-300 dark:bg-sky-950 dark:text-sky-200 dark:border-sky-700",
  },
  DOSAGE: {
    type: "DOSAGE",
    label: "Dosage",
    shortCode: "DOSE",
    hex: "#0d9488",
    highlightClass: "bg-teal-100 text-teal-950 border-teal-500 dark:bg-teal-950/60 dark:text-teal-100 dark:border-teal-400",
    chipClass: "bg-teal-100 text-teal-900 border-teal-300 dark:bg-teal-950 dark:text-teal-200 dark:border-teal-700",
  },
  FREQUENCY: {
    type: "FREQUENCY",
    label: "Frequency",
    shortCode: "FREQ",
    hex: "#4f46e5",
    highlightClass: "bg-indigo-100 text-indigo-950 border-indigo-500 dark:bg-indigo-950/60 dark:text-indigo-100 dark:border-indigo-400",
    chipClass: "bg-indigo-100 text-indigo-900 border-indigo-300 dark:bg-indigo-950 dark:text-indigo-200 dark:border-indigo-700",
  },
  ROUTE: {
    type: "ROUTE",
    label: "Route",
    shortCode: "RTE",
    hex: "#7c3aed",
    highlightClass: "bg-violet-100 text-violet-950 border-violet-500 dark:bg-violet-950/60 dark:text-violet-100 dark:border-violet-400",
    chipClass: "bg-violet-100 text-violet-900 border-violet-300 dark:bg-violet-950 dark:text-violet-200 dark:border-violet-700",
  },
  DURATION: {
    type: "DURATION",
    label: "Duration",
    shortCode: "DUR",
    hex: "#0891b2",
    highlightClass: "bg-cyan-100 text-cyan-950 border-cyan-500 dark:bg-cyan-950/60 dark:text-cyan-100 dark:border-cyan-400",
    chipClass: "bg-cyan-100 text-cyan-900 border-cyan-300 dark:bg-cyan-950 dark:text-cyan-200 dark:border-cyan-700",
  },
  LAB_TEST: {
    type: "LAB_TEST",
    label: "Lab Test",
    shortCode: "LAB",
    hex: "#059669",
    highlightClass: "bg-emerald-100 text-emerald-950 border-emerald-500 dark:bg-emerald-950/60 dark:text-emerald-100 dark:border-emerald-400",
    chipClass: "bg-emerald-100 text-emerald-900 border-emerald-300 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-700",
  },
  LAB_VALUE: {
    type: "LAB_VALUE",
    label: "Lab Value",
    shortCode: "VAL",
    hex: "#16a34a",
    highlightClass: "bg-green-100 text-green-950 border-green-500 dark:bg-green-950/60 dark:text-green-100 dark:border-green-400",
    chipClass: "bg-green-100 text-green-900 border-green-300 dark:bg-green-950 dark:text-green-200 dark:border-green-700",
  },
  LAB_UNIT: {
    type: "LAB_UNIT",
    label: "Lab Unit",
    shortCode: "UNIT",
    hex: "#65a30d",
    highlightClass: "bg-lime-100 text-lime-950 border-lime-500 dark:bg-lime-950/60 dark:text-lime-100 dark:border-lime-400",
    chipClass: "bg-lime-100 text-lime-900 border-lime-300 dark:bg-lime-950 dark:text-lime-200 dark:border-lime-700",
  },
  PROCEDURE: {
    type: "PROCEDURE",
    label: "Procedure",
    shortCode: "PROC",
    hex: "#db2777",
    highlightClass: "bg-pink-100 text-pink-950 border-pink-500 dark:bg-pink-950/60 dark:text-pink-100 dark:border-pink-400",
    chipClass: "bg-pink-100 text-pink-900 border-pink-300 dark:bg-pink-950 dark:text-pink-200 dark:border-pink-700",
  },
  ANATOMY: {
    type: "ANATOMY",
    label: "Anatomy",
    shortCode: "ANAT",
    hex: "#9333ea",
    highlightClass: "bg-purple-100 text-purple-950 border-purple-500 dark:bg-purple-950/60 dark:text-purple-100 dark:border-purple-400",
    chipClass: "bg-purple-100 text-purple-900 border-purple-300 dark:bg-purple-950 dark:text-purple-200 dark:border-purple-700",
  },
  DATE: {
    type: "DATE",
    label: "Date / Time",
    shortCode: "DATE",
    hex: "#475569",
    highlightClass: "bg-slate-200 text-slate-900 border-slate-500 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-400",
    chipClass: "bg-slate-100 text-slate-800 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600",
  },
  PATIENT_ID: {
    type: "PATIENT_ID",
    label: "Patient ID",
    shortCode: "PID",
    hex: "#64748b",
    highlightClass: "bg-zinc-200 text-zinc-900 border-zinc-500 dark:bg-zinc-800 dark:text-zinc-100 dark:border-zinc-400",
    chipClass: "bg-zinc-100 text-zinc-800 border-zinc-300 dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-600",
  },
};

const FALLBACK_TYPE_STYLE: EntityTypeStyle = {
  type: "UNKNOWN",
  label: "Entity",
  shortCode: "ENT",
  hex: "#334155",
  highlightClass: "bg-slate-100 text-slate-900 border-slate-500 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-400",
  chipClass: "bg-slate-100 text-slate-800 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600",
};

export function getEntityTypeStyle(entityType: string): EntityTypeStyle {
  return ENTITY_TYPE_STYLES[entityType] ?? { ...FALLBACK_TYPE_STYLE, type: entityType, label: entityType };
}

export function normalizeAssertion(status?: string, temporalStatus?: string): CanonicalAssertion {
  const s = (status ?? "").toUpperCase();
  if (s === "NEGATED" || s === "ABSENT") return "absent";
  if (s === "UNCERTAIN" || s === "POSSIBLE") return "possible";
  if (s === "HYPOTHETICAL" || s === "CONDITIONAL") return "hypothetical";
  if (s === "FAMILY_HISTORY" || s === "FAMILY") return "family";
  if (s === "HISTORICAL") return "historical";
  if ((temporalStatus ?? "").toUpperCase() === "HISTORICAL") return "historical";
  return "present";
}

export function getAssertionStyle(status?: string, temporalStatus?: string): AssertionStyle {
  const canonical = normalizeAssertion(status, temporalStatus);
  switch (canonical) {
    case "absent":
      return {
        canonical,
        badge: "NOT",
        showClock: false,
        textClass: "line-through",
        borderStyleClass: "border-b-2 border-solid",
        ariaLabel: "Negated (absent)",
      };
    case "possible":
      return {
        canonical,
        badge: "?",
        showClock: false,
        textClass: "",
        borderStyleClass: "border-b-2 border-dashed",
        ariaLabel: "Uncertain (possible)",
      };
    case "hypothetical":
      return {
        canonical,
        badge: "HYP",
        showClock: false,
        textClass: "italic",
        borderStyleClass: "border-b-2 border-dotted",
        ariaLabel: "Hypothetical / conditional",
      };
    case "family":
      return {
        canonical,
        badge: "FAM",
        showClock: false,
        textClass: "",
        borderStyleClass: "border-b-2 border-solid",
        ariaLabel: "Family history",
      };
    case "historical":
      return {
        canonical,
        badge: "HIST",
        showClock: true,
        textClass: "",
        borderStyleClass: "border-b-2 border-solid",
        ariaLabel: "Historical",
      };
    default:
      return {
        canonical: "present",
        badge: null,
        showClock: false,
        textClass: "",
        borderStyleClass: "border-b-2 border-solid",
        ariaLabel: "Affirmed (present)",
      };
  }
}

export function getEntityText(ent: Partial<ExtractedEntity> | undefined | null): string {
  if (!ent) return "";
  return ent.source_text ?? ent.surface_text ?? ent.normalized_text ?? "";
}
