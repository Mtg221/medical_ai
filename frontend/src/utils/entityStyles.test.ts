import { describe, expect, it } from "vitest";
import {
  ENTITY_TYPE_STYLES,
  getAssertionStyle,
  getEntityText,
  getEntityTypeStyle,
  normalizeAssertion,
} from "./entityStyles";

describe("entityStyles", () => {
  it("provides distinct colors and codes for all canonical entity types", () => {
    const types = [
      "DIAGNOSIS",
      "SYMPTOM",
      "MEDICATION",
      "DOSAGE",
      "FREQUENCY",
      "ROUTE",
      "DURATION",
      "LAB_TEST",
      "LAB_VALUE",
      "LAB_UNIT",
      "PROCEDURE",
      "ANATOMY",
    ];
    for (const t of types) {
      const style = getEntityTypeStyle(t);
      expect(style.type).toBe(t);
      expect(style.hex).toMatch(/^#/);
      expect(style.highlightClass.length).toBeGreaterThan(0);
    }
    expect(ENTITY_TYPE_STYLES.DIAGNOSIS.hex).not.toBe(ENTITY_TYPE_STYLES.MEDICATION.hex);
    expect(ENTITY_TYPE_STYLES.DOSAGE.hex).not.toBe(ENTITY_TYPE_STYLES.MEDICATION.hex);
  });

  it("maps API and canonical assertion statuses to required visual styles", () => {
    expect(normalizeAssertion("NEGATED")).toBe("absent");
    expect(normalizeAssertion("absent")).toBe("absent");
    expect(getAssertionStyle("NEGATED")).toMatchObject({
      canonical: "absent",
      badge: "NOT",
      textClass: "line-through",
    });

    expect(getAssertionStyle("UNCERTAIN")).toMatchObject({
      canonical: "possible",
      badge: "?",
      borderStyleClass: "border-b-2 border-dashed",
    });

    expect(getAssertionStyle("HYPOTHETICAL")).toMatchObject({
      canonical: "hypothetical",
      badge: "HYP",
      textClass: "italic",
    });

    expect(getAssertionStyle("FAMILY_HISTORY")).toMatchObject({
      canonical: "family",
      badge: "FAM",
    });

    expect(getAssertionStyle("AFFIRMED", "HISTORICAL")).toMatchObject({
      canonical: "historical",
      showClock: true,
    });
  });

  it("resolves source_text or surface_text safely", () => {
    expect(getEntityText({ source_text: "500 mg" })).toBe("500 mg");
    expect(getEntityText({ surface_text: "metformine" })).toBe("metformine");
    expect(getEntityText(null)).toBe("");
  });
});
