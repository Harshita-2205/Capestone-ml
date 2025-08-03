from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input/Output Models
class InputText(BaseModel):
    text: str

class ResponseMessage(BaseModel):
    response: str

# Mock functions for agents (replace with actual Mini-CDSS logic)
def talk_to_prelim_agent(input_text: str) -> str:
    """
    Simulates communication with the Prelim agent.
    Replace with the actual logic from the Mini-CDSS repository.
    """
    return f"Prelim Agent processed: {input_text}"

def talk_to_best_diag_agent(input_text: str) -> str:
    """
    Simulates communication with the BestDiag agent.
    Replace with the actual logic from the Mini-CDSS repository.
    """
    return f"BestDiag Agent processed: {input_text}"

# API Endpoints

@app.post("/input/", response_model=ResponseMessage)
def send_input_text(input_text: InputText):
    """
    Endpoint to send input text.
    """
    try:
        # Simply return the received text for now
        return {"response": f"Received input: {input_text.text}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/agent/prelim/", response_model=ResponseMessage)
def talk_with_prelim(input_text: InputText):
    """
    Endpoint to communicate with the Prelim agent.
    """
    try:
        response = talk_to_prelim_agent(input_text.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/agent/bestdiag/", response_model=ResponseMessage)
def talk_with_best_diag(input_text: InputText):
    """
    Endpoint to communicate with the BestDiag agent.
    """
    try:
        response = talk_to_best_diag_agent(input_text.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
