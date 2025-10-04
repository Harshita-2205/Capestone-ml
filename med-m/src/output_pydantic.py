from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# -------------------------------
# Existing Schemas (Clinical NER)
# -------------------------------

class NERValidationOutput(BaseModel):
    age: str = Field(..., description="The patient's age.")
    sex: str = Field(..., description="The patient's sex (e.g., Male, Female, Other).")
    history: Optional[List[str]] = Field(
        None, description="List of past medical history or conditions."
    )
    presenting_complaint: str = Field(
        ..., description="The primary complaint or issue reported by the patient."
    )
    signs_and_symptoms: List[str] = Field(
        ..., description="List of identified signs and symptoms observed or reported."
    )
    examinations_before_checkup: List[str] = Field(
        ..., description="Examinations performed on the patient."
    )
    vital_signs: Optional[Dict[str, str]] = Field(
        None,
        description="Vital signs observed, including key metrics like heart rate, blood pressure, and temperature.",
    )
    laboratory_values: Optional[Dict[str, str]] = Field(
        None,
        description="Key laboratory findings with test names as keys and results as values.",
    )
    extra_summary: Optional[str] = Field(
        None, description="Any additional summary or observations not covered by other fields."
    )


class PreliminaryDiagnosisOutput(BaseModel):
    preliminary_diagnosis: str = Field(
        ..., description="Potential health concern identified based on the input data."
    )
    reasoning: str = Field(
        ..., description="Explanation linking the symptoms, history, and observations to the diagnosis."
    )
    recommendations: List[str] = Field(
        ..., description="Suggested actions such as further tests, treatments, or referrals."
    )


class PreliminaryDiagnosisListOutput(BaseModel):
    entries: List[PreliminaryDiagnosisOutput] = Field(
        ..., description="List of three preliminary diagnosis outputs."
    )


# -------------------------------
# New Schemas (Medicine RAG + NER)
# -------------------------------

class MedicineInfoOutput(BaseModel):
    """Information about a medicine extracted from RAG or NER."""
    brand_name: str = Field(..., description="Brand name of the medicine.")
    generic_name: str = Field(..., description="Generic/chemical name of the medicine.")
    composition: List[str] = Field(..., description="Active ingredients in the medicine.")
    uses: List[str] = Field(..., description="Conditions or diseases the medicine is used for.")
    side_effects: Optional[List[str]] = Field(
        None, description="Possible side effects if known."
    )
    precautions: Optional[List[str]] = Field(
        None, description="Precautions or contraindications for the medicine."
    )


class MedicineComparisonOutput(BaseModel):
    """Comparison between prescribed and available medicines."""
    prescribed: MedicineInfoOutput = Field(..., description="Details of the originally prescribed medicine.")
    alternative: MedicineInfoOutput = Field(..., description="Details of the suggested alternative medicine.")
    are_equivalent: bool = Field(
        ..., description="Whether the alternative medicine has equivalent composition and can be substituted safely."
    )
    reasoning: str = Field(
        ..., description="Explanation of the comparison decision (composition match, therapeutic equivalence, etc)."
    )
