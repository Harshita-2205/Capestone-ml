from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import crews and helper functions
from src.main_agents import (
    ner_validation_crew,
    prelim_diag_crew,
    report_writing_crew,
)
from src.medicine_rag_agent import (
    answer_query,
    get_medicine_info,
    compare_medicines,
)

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI()

# CORS setup (loosened for dev, tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Pydantic Models
# -------------------------
class InputText(BaseModel):
    text: str

class ResponseMessage(BaseModel):
    response: str

class MedicineQuery(BaseModel):
    query: str

class MedicineInfoRequest(BaseModel):
    medicine_name: str

class MedicineComparisonRequest(BaseModel):
    original_medicine: str
    alternative_medicine: str

# -------------------------
# Agent Integration Endpoints
# -------------------------

@app.post("/agent/ner/", response_model=ResponseMessage)
def run_ner(input_text: InputText):
    """Run NER validation crew."""
    try:
        result = ner_validation_crew.kickoff(inputs={
            "input_text": input_text.text,
            "ner_output": "{}",  # TODO: plug real NER output here
        })
        return {"response": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER Error: {str(e)}")


@app.post("/agent/prelim/", response_model=ResponseMessage)
def run_prelim(input_text: InputText):
    """Run preliminary diagnosis crew."""
    try:
        result = prelim_diag_crew.kickoff(inputs={
            "post_ner_report": input_text.text,
            "output_count": 3,
        })
        return {"response": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prelim Error: {str(e)}")


@app.post("/agent/report/", response_model=ResponseMessage)
def run_report(input_text: InputText):
    """Generate structured clinical report."""
    try:
        result = report_writing_crew.kickoff(inputs={
            "post_ner_report": input_text.text,
            "prelim_diagnosis": "Sample Prelim Diagnosis",
            "best_pracs": "Sample Best Practices",
        })
        return {"response": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report Error: {str(e)}")


# -------------------------
# Medicine RAG Endpoints
# -------------------------

@app.post("/medicine/query", response_model=ResponseMessage)
def rag_freeform(med_query: MedicineQuery):
    """Freeform medical Q&A with RAG pipeline."""
    try:
        result = answer_query(med_query.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")


@app.post("/medicine/info")
def medicine_info(req: MedicineInfoRequest):
    """Structured medicine information lookup."""
    try:
        result = get_medicine_info(req.medicine_name)
        return result.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MedicineInfo Error: {str(e)}")


@app.post("/medicine/compare")
def medicine_compare(req: MedicineComparisonRequest):
    """Compare two medicines with structured output."""
    try:
        result = compare_medicines(
            req.original_medicine,
            req.alternative_medicine
        )
        return result.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MedicineCompare Error: {str(e)}")
