from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class NERValidationOutput(BaseModel):
    age: str = Field(..., description="The patient's age.")
    sex: str = Field(..., description="The patient's sex (e.g., Male, Female, Other).")
    history: Optional[List[str]] = Field(
        None, 
        description="List of past medical history or conditions."
    )
    presenting_complaint: str = Field(
        ..., 
        description="The primary complaint or issue reported by the patient."
    )
    signs_and_symptoms: List[str] = Field(
        ..., 
        description="List of identified signs and symptoms observed or reported."
    )
    examinations_before_checkup: List[str] = Field(
        ..., 
        description="Examinations performed on the patient."
    )
    vital_signs: Optional[Dict[str,str]] = Field(
        None,
        description="Vital signs observed, including key metrics like heart rate, blood pressure, and temperature."
    )
    laboratory_values: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Key laboratory findings with test names as keys and results as values. "
        )
    )
    extra_summary: Optional[str] = Field(
        None, 
        description="Any additional summary or observations not covered by other fields."
    )


class PreliminaryDiagnosisOutput(BaseModel):
    preliminary_diagnosis: str = Field(
        ...,
        description="A concise description of potential health concerns identified based on the input data."
    )
    reasoning: str = Field(
        ...,
        description="A detailed explanation linking the symptoms, history, and observations to the diagnosis."
    )
    recommendations: List[str] = Field(
        ...,
        description="A list of suggested actions such as further tests, treatments, or referrals."
    )

class PreliminaryDiagnosisListOutput(BaseModel):
    entries: List[PreliminaryDiagnosisOutput] = Field(
        ...,
        description="A list of three preliminary diagnosis outputs, each with diagnosis, reasoning, and recommendations."
    )


