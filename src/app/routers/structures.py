import json
import logging
import re
from typing import Any, Dict
from src.app.routers.agents.cardiologist import run_cardiological_debate, run_cardiological_diagnosis, run_cardiological_improved_diagnosis
from src.app.routers.agents.ophthalmologist import run_ophthalmological_debate, run_ophthalmological_diagnosis, run_ophthalmological_improved_diagnosis
from fastapi import APIRouter, Body, FastAPI, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from src.app.models import CardiologistHistory, Case, FollowUpResponse, FollowUpResponseSpecialists, NeurologistHistory, OphthalmologistHistory
from src.app.routers.agents.neurologist import run_neurological_debate, run_neurological_diagnosis, run_neurological_improved_diagnosis
from datetime import datetime
from src.utils.utilities import get_db, get_llm, parse_follow_ups, parse_specialist_response

router = APIRouter(prefix="/structures")
logger=logging.getLogger(__name__)
app = FastAPI(
    title="Structure for Specialists",
    description="Handles all the flow of specialists",
    version="2.0"
)
@router.post("/specialists/initial_round")
async def initial_round(
    case_id: str = Form(...),
    model: str = Form(...),
    db: Session = Depends(get_db),
):
    """Specialists agents initial diagnosis"""
    check_case = db.query(Case).filter(Case.case_id == case_id).first()
    if not check_case:
        raise HTTPException(status_code=404, detail="Case not found")
    message=db.query(Case.user_message).filter(Case.case_id == case_id).scalar()
    followup_history = (db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar())
    files_content = db.query(Case.files_content).filter(Case.case_id == case_id).scalar()
    files_content = str(files_content) if files_content is not None else ""
    specialists_required = db.query(Case.specialists_required).filter(Case.case_id == case_id).scalar()
    if specialists_required is None:
        return None
    input_data = (
        f"Patient message:\n{message}\n\n"
        f"Attached reports:\n{files_content}"
        f"\n\nFollow-up history:\n{followup_history}"
    )
    logger.info(f"Calling neurologist for case {case_id}")
    stage="initial"
    if "Neurologist" in specialists_required:
        neuro_response = await run_neurological_diagnosis(message,followup_history,files_content, model,stage)
        neuro_entry = NeurologistHistory(
        case_id=case_id,
        user_input=input_data,
        agent_response=neuro_response,
        answered_followups=[],
        pending_questions=[],
        timestamp=datetime.utcnow(),
        )
        db.add(neuro_entry)
        db.commit()
        db.refresh(neuro_entry)
    if "Cardiologist" in specialists_required:
        cardio_response = await run_cardiological_diagnosis(message,followup_history,files_content, model,stage)
        cardio_entry = CardiologistHistory(
            case_id=case_id,
            user_input=input_data,
            agent_response=cardio_response,
            answered_followups=[],
            pending_questions=[],
            timestamp=datetime.utcnow(),
        )
        db.add(cardio_entry)
        db.commit()
        db.refresh(cardio_entry)
    if "Ophthalmologist" in specialists_required:
        ophthal_response = await run_ophthalmological_diagnosis(message,followup_history,files_content, model,stage)
        ophthal_entry = OphthalmologistHistory(
            case_id=case_id,
            user_input=input_data,
            agent_response=ophthal_response,
            answered_followups=[],
            pending_questions=[],
            timestamp=datetime.utcnow(),
        )
        db.add(ophthal_entry)
        db.commit()
        db.refresh(ophthal_entry)
    db.query(Case).filter(Case.case_id == case_id).update(
        {Case.stage: "first_debate"}, synchronize_session=False
    )
    db.commit()
    return {
        "case_id": case_id,
        "responses":{"neurologist":neuro_entry.agent_response if "Neurologist" in specialists_required else None,"cardiologist":cardio_entry.agent_response if "Cardiologist" in specialists_required else None,"ophthalmologist":ophthal_entry.agent_response if "Ophthalmologist" in specialists_required else None}
    }

def _build_history(case_id: str, agent: str, db):
    """
    Builds chat history for a given agent and returns it 
    in a LangChain-compatible format with timestamps.
    Includes answered follow-ups (Q/A pairs) as part of the history.
    """

    agent_models = {
        "neurologist": NeurologistHistory,
        "cardiologist": CardiologistHistory,
        "ophthalmologist": OphthalmologistHistory,
    }

    model = agent_models.get(agent)
    if not model:
        raise ValueError(f"Unsupported agent type: {agent}")

    records = (
        db.query(
            model.user_input,
            model.agent_response,
            model.answered_followups,
            model.timestamp,
        )
        .filter(model.case_id == case_id)
        .order_by(model.timestamp.asc())
        .all()
    )

    history = []
    for record in records:
        user_input, agent_response, answered_followups, timestamp = record
        ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp

        if user_input:
            history.append({
                "role": "user",
                "content": user_input,
                "timestamp": ts,
            })

        if answered_followups:
            for qa in answered_followups:
                question = qa.get("question")
                answer = qa.get("answer")

                if question:
                    history.append({
                        "role": "assistant",
                        "content": f"Follow-up: {question}",
                        "timestamp": ts,
                    })
                if answer:
                    history.append({
                        "role": "user",
                        "content": answer,
                        "timestamp": ts,
                    })

        if agent_response:
            history.append({
                "role": "assistant",
                "content": agent_response,
                "timestamp": ts,
            })

    return history


def _add_follow_ups_for_db(neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate,neurologist_follow_ups,cardiologist_follow_ups,ophthalmologist_follow_ups, specialists_required):
    """Add the follow up questions asked by specialists to the database"""
    if "Neurologist" in specialists_required:
        neurologist_response_after_debate=neurologist_response_after_debate.dict()
        neurologist_response_after_debate["follow_ups"]=neurologist_follow_ups
    if "Cardiologist" in specialists_required:
        cardiologist_response_after_debate=cardiologist_response_after_debate.dict()
        cardiologist_response_after_debate["follow_ups"]=cardiologist_follow_ups
    if "Ophthalmologist" in specialists_required:
        ophthalmologist_response_after_debate=ophthalmologist_response_after_debate.dict()
        ophthalmologist_response_after_debate["follow_ups"]=ophthalmologist_follow_ups
    return neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate

def _common_followups(neurologist_follow_ups, cardiologist_follow_ups, ophthalmologist_follow_ups, model, specialists_required)->list:
    """Returns unique combined follow-up questions from specialists, excluding duplicates and (optionally) those similar to GP questions."""
    llm = get_llm(model)
    system_prompt = """Your task is to combine the follow-up questions from different specialists into a unique list, removing any duplicates or very similar questions.
    
    Just give me the unique follow-up questions comma-separated without any additional text.
    Output Format:
    follow_ups: <question1, question2, ...>
    If none, follow_ups: None
    """
    all_specialist = f"Neurologist: {neurologist_follow_ups}\nCardiologist: {cardiologist_follow_ups}\nOphthalmologist: {ophthalmologist_follow_ups}"
    
    prompt = f"{system_prompt}\n\n{all_specialist}"
    try:
        raw_response = llm.invoke(prompt).content.strip()
        logger.info(f"LLM Raw response: {raw_response}")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="LLM backend error")
    follow_ups = parse_follow_ups(raw_response)
    return follow_ups

@router.post("/specialists/debate_round")
async def debate_round(
    case_id: str = Form(...),
    model: str = Form(...),
    #initial_responses: str | dict = Form(...),
    db: Session = Depends(get_db),
):
    """Specialist agents will debate on the diagnosis of each."""
    # if isinstance(initial_responses, str):
    #     try:
    #         initial_responses = json.loads(initial_responses)
    #     except json.JSONDecodeError:
    #         raise HTTPException(status_code=400, detail="Invalid JSON in initial_responses")
    # elif not isinstance(initial_responses, dict):
    #     raise HTTPException(status_code=400, detail="initial_responses must be a dict or JSON string")

    check_case = db.query(Case).filter(Case.case_id == case_id).first()
    if not check_case:
        raise HTTPException(status_code=404, detail="Case not found")
    # follow up history not needed because already stored in database for each agent's chat history.
    #followup_history = (db.query(Case.answered_followups).filter(Case.case_id == case_id).scalar())
    specialists_required = db.query(Case.specialists_required).filter(Case.case_id == case_id).scalar()
    initial_responses={}
    for specialist in specialists_required:
        if specialist == "Neurologist":
            initial_responses["neurologist"] = db.query(NeurologistHistory).filter(NeurologistHistory.case_id == case_id).order_by(NeurologistHistory.timestamp.desc()).first().agent_response
        elif specialist == "Cardiologist":
            initial_responses["cardiologist"] = db.query(CardiologistHistory).filter(CardiologistHistory.case_id == case_id).order_by(CardiologistHistory.timestamp.desc()).first().agent_response
        elif specialist == "Ophthalmologist":
            initial_responses["ophthalmologist"] = db.query(OphthalmologistHistory).filter(OphthalmologistHistory.case_id == case_id).order_by(OphthalmologistHistory.timestamp.desc()).first().agent_response


    neurologist_response = "Neurologist was not involved in this case."
    cardiologist_response = "Cardiologist was not involved in this case."
    ophthalmologist_response = "Ophthalmologist was not involved in this case."
    neurologist_response_after_debate = ""
    cardiologist_response_after_debate = ""
    ophthalmologist_response_after_debate = ""
    neurologist_follow_ups=[]
    cardiologist_follow_ups=[]
    ophthalmologist_follow_ups=[]
    neurologist_rag = ""
    cardiologist_rag = ""
    ophthalmologist_rag = ""
    neurologist_history = ""
    cardiologist_history = ""
    ophthalmologist_history = ""
    check_flag = None # used to track the follow ups as they are common. Flag is used to track for accessing the database 

    if specialists_required is None:
        return None
    if "Neurologist" in specialists_required:
        neurologist_response = initial_responses["neurologist"]
        neurologist_rag = None # TODO: RAG for neurologist
        neurologist_history = _build_history(case_id,"neurologist",db)
        check_flag = "Neurologist"
    if "Cardiologist" in specialists_required:
        cardiologist_response = initial_responses["cardiologist"]
        cardiologist_rag = None # TODO: RAG for cardiologist
        cardiologist_history = _build_history(case_id,"cardiologist",db)
        check_flag = "Cardiologist"
    if "Ophthalmologist" in specialists_required:
        ophthalmologist_response = initial_responses["ophthalmologist"]
        ophthalmologist_rag = None # TODO: RAG for ophthalmologist
        ophthalmologist_history = _build_history(case_id,"ophthalmologist",db)
        check_flag = "Ophthalmologist"

    # Starting debate

    if "Neurologist" in specialists_required:
        neurologist_response_after_debate, neurologist_debate_prompt, neurologist_follow_ups = await run_neurological_debate(case_id,neurologist_history,cardiologist_response,ophthalmologist_response,neurologist_rag,model)
    
    if "Cardiologist" in specialists_required:
        cardiologist_response_after_debate, cardiologist_debate_prompt, cardiologist_follow_ups = await run_cardiological_debate(case_id,cardiologist_history,neurologist_response,ophthalmologist_response,cardiologist_rag,model)
    
    if "Ophthalmologist" in specialists_required:
        ophthalmologist_response_after_debate, ophthalmologist_debate_prompt, ophthalmologist_follow_ups = await run_ophthalmological_debate(case_id,ophthalmologist_history,neurologist_response,cardiologist_response,ophthalmologist_rag,model)
    
    if "Neurologist" in specialists_required:
        neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate = _add_follow_ups_for_db(neurologist_response_after_debate,cardiologist_response_after_debate,ophthalmologist_response_after_debate,neurologist_follow_ups,cardiologist_follow_ups,ophthalmologist_follow_ups, specialists_required)
        common_follow_ups=_common_followups(neurologist_follow_ups,cardiologist_follow_ups,ophthalmologist_follow_ups,model, specialists_required)
        neuro_entry = NeurologistHistory(
            case_id=case_id,
            user_input=neurologist_debate_prompt,
            agent_response=neurologist_response_after_debate,
            answered_followups=[],
            pending_questions=common_follow_ups,
            timestamp=datetime.utcnow(),
        )
        db.add(neuro_entry)
        db.commit()
        db.refresh(neuro_entry)
    if "Cardiologist" in specialists_required:
        cardio_entry = CardiologistHistory(
            case_id=case_id,
            user_input=cardiologist_debate_prompt,
            agent_response=cardiologist_response_after_debate,
            answered_followups=[],
            pending_questions=common_follow_ups,
            timestamp=datetime.utcnow(),
        )
        db.add(cardio_entry)
        db.commit()
        db.refresh(cardio_entry)

    if "Ophthalmologist" in specialists_required:
        ophthal_entry = OphthalmologistHistory(
            case_id=case_id,
            user_input=ophthalmologist_debate_prompt,
            agent_response=ophthalmologist_response_after_debate,
            answered_followups=[],
            pending_questions=common_follow_ups,
            timestamp=datetime.utcnow(),
        )
        db.add(ophthal_entry)
        db.commit()
        db.refresh(ophthal_entry)
    if common_follow_ups:
        db.query(Case).filter(Case.case_id == case_id).update(
            {Case.stage: "follow_up_specialist"}, synchronize_session=False
        )
        db.commit()
    else:
        db.query(Case).filter(Case.case_id == case_id).update(
            {Case.stage: "improved_diagnosis"}, synchronize_session=False
        )
        db.commit()
    
    return {
        case_id:case_id,
        "responses":{"neurologist":neurologist_response_after_debate if "Neurologist" in specialists_required else None,"cardiologist":cardiologist_response_after_debate if "Cardiologist" in specialists_required else None,"ophthalmologist":ophthalmologist_response_after_debate if "Ophthalmologist" in specialists_required else None},
        "follow_ups":{"neurologist":neurologist_follow_ups if "Neurologist" in specialists_required else None,"cardiologist":cardiologist_follow_ups if "Cardiologist" in specialists_required else None,"ophthalmologist":ophthalmologist_follow_ups if "Ophthalmologist" in specialists_required else None},
        "follow_ups_common":(check_flag, common_follow_ups[0]),
    } 


@router.post("/specialists/answer_followup", response_model=FollowUpResponseSpecialists)
async def answer_followup(
    case_id: str = Form(..., description="Case ID of the ongoing consultation"),
    answer: str = Form(..., description="Patient's answer to the current follow-up question"),
    db: Session = Depends(get_db),
    check_flag: str = Form(..., description="Flag to check which table to access"),
):
    """
    Handles patient answers to follow-up questions.
    Moves through pending questions until all are answered.
    """
    if check_flag == "Neurologist":
        specialists_table_name = NeurologistHistory
    elif check_flag == "Cardiologist":
        specialists_table_name = CardiologistHistory
    elif check_flag == "Ophthalmologist":
        specialists_table_name = OphthalmologistHistory
    case=db.query(Case).filter(Case.case_id == case_id).first()
    specialists_required = db.query(Case.specialists_required).filter(Case.case_id == case_id).scalar()
    specialists_table = db.query(specialists_table_name).filter(specialists_table_name.case_id == case_id).first()
    if not specialists_table:
        raise HTTPException(status_code=404, detail="Case ID not found")
    specialists_table = db.query(specialists_table_name).filter(specialists_table_name.case_id == case_id).order_by(specialists_table_name.timestamp.desc()).first()
    logging.info(f"Case {specialists_table}")
    if not specialists_table.pending_questions:
        pending_questions = specialists_table.pending_questions
        answered_followups = specialists_table.answered_followups
        for specialist in specialists_required:
            if specialist != check_flag:
                if specialist == "Neurologist":
                    specialists_table_name = NeurologistHistory
                elif specialist == "Cardiologist":
                    specialists_table_name = CardiologistHistory
                elif specialist == "Ophthalmologist":
                    specialists_table_name = OphthalmologistHistory
                specialists_table_name.answered_followups = answered_followups
                specialists_table_name.pending_questions = pending_questions
                db.commit()
                db.refresh(specialists_table_name)
            
        db.query(Case).filter(Case.case_id == case_id).update(
            {Case.stage: "improved_diagnosis"}, synchronize_session=False
        )
        db.commit()

        return FollowUpResponseSpecialists(
            case_id=specialists_table.case_id,
            stage="completed",
            message="No pending follow-up questions.",
            next_followup=None,
            answered_followups=specialists_table.answered_followups
        )

    current_question = specialists_table.pending_questions[0]
    remaining_questions = specialists_table.pending_questions[1:]
    specialists_table.pending_questions = remaining_questions  
    if not specialists_table.answered_followups:
        specialists_table.answered_followups = []
    specialists_table.answered_followups = specialists_table.answered_followups + [
        {"question": current_question, "answer": answer}
    ]
    db.commit()
    db.refresh(specialists_table)

    if specialists_table.pending_questions:
        next_question = specialists_table.pending_questions[0]

        return FollowUpResponseSpecialists(
            case_id=specialists_table.case_id,
            stage="follow_up",
            message="Answer recorded.",
            next_followup=next_question,
            answered_followups=specialists_table.answered_followups
        )
    else:
        next_action_message = "All GP follow-up questions answered. Case completed."

        db.commit()
        db.refresh(specialists_table)

        return FollowUpResponseSpecialists(
            case_id=specialists_table.case_id,
            stage="second_debate",
            message=next_action_message,
            next_followup=None,
            answered_followups=specialists_table.answered_followups
        )


@router.get("/specialists/get_case_state/{case_id}", response_model=FollowUpResponseSpecialists)
async def get_case_state(case_id: str, db: Session = Depends(get_db)):
    """
    Get the current state of a case for debugging purposes.
    """
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found")

    next_followup = case.pending_questions[0] if case.pending_questions else None

    return FollowUpResponseSpecialists(
        case_id=case.case_id,
        message="Debug info: current case state.",
        next_followup=next_followup,
        answered_followups=case.answered_followups
    )


@router.post("/specialists/improved_diagnosis")
async def specialists_improved_diagnosis(
    case_id: str = Form(..., description="Case ID of the ongoing consultation"),
    model: str = Form(..., description="Backend model: 'gpt' or 'gemini'"),
    db: Session = Depends(get_db),
):
    """
    each specialist refines their diagnosis
    after reviewing follow-up answers and prior debates.
    Saves the improved diagnosis and updates case stage to 'completed'.
    """
    check_case = db.query(Case).filter(Case.case_id == case_id).first()
    if not check_case:
        raise HTTPException(status_code=404, detail="Case not found")

    specialists_required = db.query(Case.specialists_required).filter(Case.case_id == case_id).scalar()
    if not specialists_required:
        raise HTTPException(status_code=400, detail="No specialists assigned for this case")

    neurologist_history = cardiologist_history = ophthalmologist_history = None
    if "Neurologist" in specialists_required:
        neurologist_history = _build_history(case_id, "neurologist", db)
    if "Cardiologist" in specialists_required:
        cardiologist_history = _build_history(case_id, "cardiologist", db)
    if "Ophthalmologist" in specialists_required:
        ophthalmologist_history = _build_history(case_id, "ophthalmologist", db)

    neurologist_final = cardiologist_final = ophthalmologist_final = None

    try:
        if "Neurologist" in specialists_required:
            neurologist_final, neurological_user_input = await run_neurological_improved_diagnosis(
                case_id=case_id,
                history=neurologist_history,
                model=model,
            )
            neuro_entry = NeurologistHistory(
                case_id=case_id,
                user_input=neurological_user_input,
                agent_response={
                    "confidence": neurologist_final.confidence,
                    "diagnosis": neurologist_final.diagnosis,
                    "explanation": neurologist_final.explanation
                },
                answered_followups=[],
                pending_questions=[],
                timestamp=datetime.utcnow(),
            )
            db.add(neuro_entry)
            db.commit()
            db.refresh(neuro_entry)

        if "Cardiologist" in specialists_required:
            cardiologist_final, cardiological_user_input = await run_cardiological_improved_diagnosis(
                case_id=case_id,
                history=cardiologist_history,
                model=model,
            )
            cardio_entry = CardiologistHistory(
                case_id=case_id,
                user_input=cardiological_user_input,
                agent_response={
                    "confidence": cardiologist_final.confidence,
                    "diagnosis": cardiologist_final.diagnosis,
                    "explanation": cardiologist_final.explanation
                },
                answered_followups=[],
                pending_questions=[],
                timestamp=datetime.utcnow(),
            )
            db.add(cardio_entry)
            db.commit()
            db.refresh(cardio_entry)

        if "Ophthalmologist" in specialists_required:
            ophthalmologist_final, ophthalmological_user_input = await run_ophthalmological_improved_diagnosis(
                case_id=case_id,
                history=ophthalmologist_history,
                model=model,
            )
            ophthal_entry = OphthalmologistHistory(
                case_id=case_id,
                user_input=ophthalmological_user_input,
                agent_response={
                    "confidence": ophthalmologist_final.confidence,
                    "diagnosis": ophthalmologist_final.diagnosis,
                    "explanation": ophthalmologist_final.explanation
                },
                answered_followups=[],
                pending_questions=[],
                timestamp=datetime.utcnow(),
            )
            db.add(ophthal_entry)
            db.commit()
            db.refresh(ophthal_entry)

    except Exception as e:
        logger.error(f"Error during improved diagnosis for case {case_id}: {e}")
        raise HTTPException(status_code=500, detail="Error running improved diagnosis")

    db.query(Case).filter(Case.case_id == case_id).update(
        {Case.stage: "second_debate"}, synchronize_session=False
    )
    db.commit()

    return {
        "case_id": case_id,
        "status": "success",
        "message": "Improved diagnosis completed for all assigned specialists.",
        "results": {
            "neurologist": neurologist_final.dict() if neurologist_final else None,
            "cardiologist": cardiologist_final.dict() if cardiologist_final else None,
            "ophthalmologist": ophthalmologist_final.dict() if ophthalmologist_final else None,
        }
    }

def _parse_consensus_output(raw_output: str) -> Dict[str, Any]:
    """
    Parse the raw LLM output for consensus winner into a structured dict:
    {
    "winner": <specialist_name>,
    "diagnosis": <diagnosis>,
    "explanation": <explanation>
    }

    Handles plain text where fields are labeled in the format:
    winner <specialist_name>
    diagnosis <diagnosis text>
    explanation <explanation text>
    """
    result = {"winner": None, "diagnosis": None, "explanation": None}

    winner_match = re.search(r"winner\s*:\s*(.+)", raw_output, re.IGNORECASE)
    diagnosis_match = re.search(r"diagnosis\s*:\s*(.+)", raw_output, re.IGNORECASE | re.DOTALL)
    explanation_match = re.search(r"explanation\s*:\s*(.+)", raw_output, re.IGNORECASE | re.DOTALL)


    if winner_match:
        result["winner"] = winner_match.group(1).strip()

    if diagnosis_match:
        result["diagnosis"] = diagnosis_match.group(1).strip()

    if explanation_match:
        # Note: If explanation is the last field, the regex will capture everything to the end.
        result["explanation"] = explanation_match.group(1).strip()

    return result

@router.post("/specialists/determine_consensus_winner")
async def determine_consensus_winner(case_id: str, model: str, db: Session = Depends(get_db)):
    """
    Determines the winning specialist diagnosis based on confidence.
    If there's a tie, uses LLM to decide.
    Returns: {"winner": <specialist>, "diagnosis": <str>, "explanation": <str>}
    """

    specialists_required = db.query(Case.specialists_required).filter(Case.case_id == case_id).scalar()
    if not specialists_required:
        raise HTTPException(status_code=400, detail="No specialists assigned for this case")

    agent_models = {
        "Neurologist": NeurologistHistory,
        "Cardiologist": CardiologistHistory,
        "Ophthalmologist": OphthalmologistHistory,
    }

    responses = {}
    for specialist in specialists_required:
        model_class = agent_models[specialist]
        last_entry = db.query(model_class).filter(model_class.case_id == case_id).order_by(model_class.timestamp.desc()).first()
        if last_entry and last_entry.agent_response:
            responses[specialist] = last_entry.agent_response

    if not responses:
        raise HTTPException(status_code=400, detail="No specialist responses found")

    max_confidence = max(res["confidence"] for res in responses.values() if res and "confidence" in res)
    top_specialists = [spec for spec, res in responses.items() if res["confidence"] == max_confidence]

    if len(top_specialists) == 1:
        winner = top_specialists[0]
        return {
            "winner": winner,
            "diagnosis": responses[winner]["diagnosis"],
            "explanation": responses[winner]["explanation"]
        }

    llm = get_llm(model) 
    history_prompt = ""
    for spec in specialists_required:
        if spec in responses:
            history_prompt += (
                f"{spec}:\n"
                f"Diagnosis: {responses[spec]['diagnosis']}\n"
                f"Confidence: {responses[spec]['confidence']}\n"
                f"Explanation: {responses[spec]['explanation']}\n\n"
            )

    tie_breaker_prompt = f"""
You are a medical consensus agent. Multiple specialists have provided their diagnoses with confidence scores.
Decide which specialist's diagnosis is most reliable. Always return exactly in this format:
Output Format (exactly, with no deviations) dont use ``` or other markdown:
winner: <specialist name>
diagnosis: <diagnosis>
explanation: <explanation>


Specialists responses:
{history_prompt}
"""

    llm_response_raw = llm.invoke(tie_breaker_prompt).content.strip()
    
    try:
        parsed_result = _parse_consensus_output(llm_response_raw)
        
        if not parsed_result.get("winner") or not parsed_result.get("diagnosis"):
             raise ValueError("Parsed output missing winner or diagnosis fields.")
        
        return parsed_result

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"LLM tie-breaker returned invalid response or parsing failed. Raw: '{llm_response_raw}'. Error: {e}"
        )


@router.post("/specialists/chat")
async def chat_with_agent(
    case_id: str = Form(..., description="Case ID"),
    model: str = Form(..., description="LLM model: 'gpt' or 'gemini'"),
    agent_name: str = Form(..., description="Specialist name: Neurologist, Cardiologist, Ophthalmologist"),
    chat_type: int = Form(1, description="0 for Recommendation, 1 for Chat/Assistance"),
    user_message: str = Form(None, description="User's message for chat (only required if chat_type=1)"),
    consensus_data: Dict[str, Any] = Body(None, description="Optional: Final consensus winner data {'winner', 'diagnosis', 'explanation'}"),
    db: Session = Depends(get_db),
):
    """
    Handles chat/assistance with a specific agent or provides final recommendations.
    
    If chat_type=1: Allows user to chat directly with a specialist agent.
    If chat_type=0: Provides user-friendly recommendations based on the consensus winner.
    """
    AGENT_MODELS = {
        "Neurologist": NeurologistHistory,
        "Cardiologist": CardiologistHistory,
        "Ophthalmologist": OphthalmologistHistory,
    }

    SYSTEM_PROMPTS = {
        "chat": {
            "Neurologist": "You are a professional Neurologist. Your task is to communicate with the user/patient and provide assistance and answer their questions based on the case history. Be empathetic and helpful.",
            "Cardiologist": "You are a professional Cardiologist. Your task is to communicate with the user/patient and provide assistance and answer their questions based on the case history. Be empathetic and helpful.",
            "Ophthalmologist": "You are a professional Ophthalmologist. Your task is to communicate with the user/patient and provide assistance and answer their questions based on the case history. Be empathetic and helpful.",
        },
        "recommendation": "You are a patient-facing medical communicator. Your task is to summarize the findings, diagnosis, and provide clear, user-friendly recommendations for the next steps (cure, further testing, lifestyle changes, etc.) to the patient. Use the provided FINAL CONSENSUS and the CHAT HISTORY of the winning specialist for context. Maintain an encouraging and clear tone. Do not mention confidence scores or internal specialist debates.",
    }

    if agent_name not in AGENT_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid agent name: **{agent_name}**. Must be one of: {', '.join(AGENT_MODELS.keys())}")

    AgentHistoryModel = AGENT_MODELS[agent_name]
    llm = get_llm(model)
    
    
    # --- CHAT / ASSISTANCE SCENARIO (chat_type=1) ---
    if chat_type == 1:
        if not user_message:
            raise HTTPException(status_code=400, detail="**User message** is required when `chat_type` is set to 1 (Chat/Assistance).")
            
        try:
            history_records = _build_history(case_id, agent_name.lower(), db) 
        except Exception as e:
            logger.error(f"Error building history for case {case_id} and agent {agent_name}: {e}")
            raise HTTPException(status_code=500, detail="Could not retrieve case history.")

        system_prompt = SYSTEM_PROMPTS["chat"][agent_name]
        
        full_chat_history = [{"role": "system", "content": system_prompt}]
        
        for entry in history_records:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            timestamp = entry.get("timestamp", "")
            
            formatted_content = f"[{timestamp}] {content}" if timestamp else content

            if role == "user":
                full_chat_history.append({"role": "user", "content": formatted_content})
            elif role == "assistant":
                full_chat_history.append({"role": "assistant", "content": formatted_content})
            else:
                full_chat_history.append({"role": "user", "content": formatted_content})

        full_chat_history.append({"role": "user", "content": user_message})
        
        try:
            llm_response = llm.invoke(full_chat_history).content
        except Exception as e:
            logger.error(f"LLM chat error for case {case_id}: {e}")
            raise HTTPException(status_code=500, detail="LLM chat backend error.")

        chat_entry = AgentHistoryModel(
            case_id=case_id,
            user_input=user_message,
            agent_response=llm_response,
            answered_followups=[],
            pending_questions=[],
            timestamp=datetime.utcnow(),
        )
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)

        return {
            "case_id": case_id,
            "agent_name": agent_name,
            "role": "assistant",
            "message": llm_response
        }

    # --- RECOMMENDATION SCENARIO (chat_type=0) ---
    elif chat_type == 0:
        consensus_winner_data = consensus_data
        
        if not consensus_winner_data:
            try:
                consensus_winner_data = await determine_consensus_winner(case_id, model, db)
            except HTTPException as e:
                raise HTTPException(status_code=e.status_code, detail="Failed to retrieve final consensus winner. Provide `consensus_data` or ensure the improved diagnosis stage is complete.")
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to retrieve final consensus winner.")

        if not all(k in consensus_winner_data for k in ["winner", "diagnosis", "explanation"]):
            raise HTTPException(status_code=400, detail="`consensus_data` must contain 'winner', 'diagnosis', and 'explanation'.")

        winner = consensus_winner_data["winner"]
        diagnosis = consensus_winner_data["diagnosis"]
        explanation = consensus_winner_data["explanation"]
        
        try:
            winner_history_records = _build_history(case_id, winner.lower(), db)
            winner_chat_history_list = []
            for record in winner_history_records:
                role = record.get("role", "user")
                content = record.get("content", "")
                timestamp = record.get("timestamp", "")
                
                formatted_content = f"[{timestamp}] {content}" if timestamp else content
                winner_chat_history_list.append(f"{role.upper()}: {formatted_content}")

            winner_chat_history_str = "\n".join(winner_chat_history_list)
        except Exception as e:
            logger.warning(f"Could not retrieve full history for winner {winner}: {e}")
            winner_chat_history_str = "History could not be retrieved."


        recommendation_input = (
            f"Winning Specialist: {winner}\n"
            f"Diagnosis: {diagnosis}\n"
            f"Explanation/Reasoning: {explanation}"
        )
        
        system_prompt = SYSTEM_PROMPTS["recommendation"]
        
        recommendation_prompt = f"""
{system_prompt}

--- CHAT HISTORY of the Winning Specialist ({winner}) ---
{winner_chat_history_str}

--- FINAL CONSENSUS ---
{recommendation_input}
"""

        try:
            llm_response = llm.invoke(recommendation_prompt).content
        except Exception as e:
            logger.error(f"LLM recommendation error for case {case_id}: {e}")
            raise HTTPException(status_code=500, detail="LLM recommendation backend error.")

        return {
            "case_id": case_id,
            "agent_name": winner,
            "role": "recommendation",
            "message": llm_response
        }

    raise HTTPException(status_code=400, detail="Invalid `chat_type` flag. Must be 0 (Recommendation) or 1 (Chat/Assistance).")


app.include_router(router)