import json
import logging
from src.app.routers.agents.cardiologist import run_cardiological_debate, run_cardiological_diagnosis
from src.app.routers.agents.ophthalmologist import run_ophthalmological_debate, run_ophthalmological_diagnosis
from fastapi import APIRouter, Body, FastAPI, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from src.app.models import CardiologistHistory, Case, FollowUpResponse, FollowUpResponseSpecialists, NeurologistHistory, OphthalmologistHistory
from src.app.routers.agents.neurologist import run_neurological_debate, run_neurological_diagnosis
from datetime import datetime
from src.utils.utilities import get_db, get_llm, parse_follow_ups, parse_specialist_response

router = APIRouter(prefix="/structures")
logger=logging.getLogger(__name__)
app = FastAPI(
    title="Structure for Specialists",
    description="Handles all the flow of specialists",
    version="2.0"
)
@router.post("/initial_round")
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

@router.post("/first_debate_round")
async def first_debate_round(
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


app.include_router(router)