from workflow import run_medguide

def test_workflow_without_api_key():
    patient = {
        "age": 25, "gender": "Female", "medical_history": "None",
        "allergies": "None", "medications": "None", "location": "Chennai",
        "symptoms": "fever and cough for two days"
    }
    result = run_medguide(patient)
    assert result["profile"]["age"] == 25
    assert "risk" in result
    assert "final_response" in result


def test_emergency_fast_path():
    patient = {
        "age": 40, "gender": "Male", "medical_history": "None",
        "allergies": "None", "medications": "None", "location": "Chennai",
        "symptoms": "severe chest pain and can't breathe"
    }
    result = run_medguide(patient)
    assert "EMERGENCY" in result["risk"].upper()
    assert "Emergency Fast-Path" in result["agent_log"][0]
