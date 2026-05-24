import os
import json
import time
import numpy as np
import random
import jwt
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types

from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, ARRAY, ForeignKey, Table
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

load_dotenv()

DATABASE_URL = "postgresql://postgres:sunil@localhost:5432/timetable_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "super-secret-timetable-key" 
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="AI Memetic Timetable Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

course_section_link = Table(
    'course_section_link', Base.metadata,
    Column('course_id', Integer, ForeignKey('courses.id'), primary_key=True),
    Column('section_id', Integer, ForeignKey('year_sections.id'), primary_key=True)
)

class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String) 
    teacher_id = Column(Integer, ForeignKey("teachers.id"), nullable=True)
    year = Column(Integer, nullable=True)
    section = Column(String, nullable=True)

class DBYearSection(Base):
    __tablename__ = "year_sections"
    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, nullable=False)
    section = Column(String, nullable=False)
    student_count = Column(Integer)
    courses = relationship("DBCourse", secondary=course_section_link, back_populates="target_sections")

class DBCourse(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, unique=True, index=True)
    course_name = Column(String)
    lectures_per_week = Column(Integer, default=3)
    labs_per_week = Column(Integer, default=0)
    software_needed = Column(ARRAY(String))
    target_sections = relationship("DBYearSection", secondary=course_section_link, back_populates="courses")

class DBTeacher(Base):
    __tablename__ = "teachers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    expertise = Column(ARRAY(String))

class DBRoom(Base):
    __tablename__ = "rooms"
    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String, unique=True)
    room_type = Column(String) 
    installed_software = Column(ARRAY(String))
    capacity = Column(Integer)

class DBSchedule(Base):
    __tablename__ = "schedules"
    id = Column(Integer, primary_key=True, index=True)
    schedule_data = Column(JSON)

class DBSubstitutionRequest(Base):
    __tablename__ = "substitution_requests"
    id = Column(Integer, primary_key=True, index=True)
    requester_username = Column(String)
    target_teacher_username = Column(String)
    time_slot = Column(Integer)
    course_code = Column(String)
    year = Column(Integer)
    section = Column(String)
    status = Column(String, default="pending")

Base.metadata.create_all(bind=engine)

def ai_enrich_and_map_rooms(courses, rooms) -> Dict[str, str]:
    mapping = {}
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        course_data = [{"code": c.course_code, "name": c.course_name, "labs": c.labs_per_week} for c in courses[:20]]
        room_data = [{"id": r.room_id, "type": r.room_type} for r in rooms[:20]]
        prompt = f"Match courses to rooms. If labs > 0, assign a Lab room. Return ONLY JSON: {{'course_code': 'room_id'}}. Courses: {course_data} Rooms: {room_data}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        mapping = json.loads(response.text)
    except Exception as e: pass

    lab_rooms = [r.room_id for r in rooms if r.room_type.lower() == 'lab']
    lec_rooms = [r.room_id for r in rooms if r.room_type.lower() != 'lab']
    if not lab_rooms: lab_rooms = [r.room_id for r in rooms]
    if not lec_rooms: lec_rooms = [r.room_id for r in rooms]

    for c in courses:
        if f"{c.course_code}_LEC" not in mapping:
            mapping[f"{c.course_code}_LEC"] = random.choice(lec_rooms)
        if c.labs_per_week > 0 and f"{c.course_code}_LAB" not in mapping:
            mapping[f"{c.course_code}_LAB"] = random.choice(lab_rooms)
            
    return mapping

class HybridGeneticSHOSolver:
    def __init__(self, class_instances, teachers, course_room_map, total_slots, days_per_week, pop_size=40, max_iter=200):
        self.courses = class_instances 
        self.teachers = teachers
        self.course_room_map = course_room_map
        self.total_slots = total_slots
        self.slots_per_day = total_slots // days_per_week
        self.num_classes = len(class_instances)
        self.pop_size = pop_size
        self.max_iter = max_iter 
        
        self.course_to_teacher = {}
        section_course_teacher_map = {}
        self.fast_eval_data = []
        
        for c_idx, c in enumerate(self.courses):
            sec_key = f"Y{c.target_section.year}-S{c.target_section.section}"
            full_sec_key = f"{sec_key}-{c.base_code}"
            
            if full_sec_key in section_course_teacher_map:
                self.course_to_teacher[c_idx] = section_course_teacher_map[full_sec_key]
            else:
                capable_teachers = [t_idx for t_idx, t in enumerate(self.teachers) if t.expertise and c.base_code in t.expertise]
                chosen_teacher_idx = random.choice(capable_teachers) if capable_teachers else random.randint(0, len(self.teachers)-1)
                self.course_to_teacher[c_idx] = chosen_teacher_idx
                section_course_teacher_map[full_sec_key] = chosen_teacher_idx

            t_name = self.teachers[self.course_to_teacher[c_idx]].name
            room_id = self.course_room_map.get(c.instance_code)
            self.fast_eval_data.append((t_name, room_id, sec_key, c.base_code))

        self.population = np.random.rand(self.pop_size, self.num_classes)
        spread_schedule = (np.arange(self.num_classes) % self.total_slots) / self.total_slots
        np.random.shuffle(spread_schedule)
        self.population[0] = spread_schedule

        self.best_schedule = None
        self.best_fitness = float('inf')
        self.best_bad_indices = []

    def calculate_fitness(self, schedule_continuous):
        discrete_slots = np.floor(schedule_continuous * self.total_slots).astype(int)
        discrete_slots = np.clip(discrete_slots, 0, self.total_slots - 1)
        
        conflicts = 0
        seen_teacher = set()
        seen_room = set()
        seen_section = set()
        seen_daily_class = set() 
        bad_indices = set()
        
        for i in range(self.num_classes):
            slot = discrete_slots[i]
            t_name, room_id, sec_key, base_code = self.fast_eval_data[i]
            
            day_of_week = slot // self.slots_per_day
            
            t_key = (t_name, slot)
            r_key = (room_id, slot)
            s_key = (sec_key, slot)
            d_key = (sec_key, base_code, day_of_week)
            
            if t_key in seen_teacher: 
                conflicts += 100; bad_indices.add(i)
            else: seen_teacher.add(t_key)
                
            if r_key in seen_room: 
                conflicts += 100; bad_indices.add(i)
            else: seen_room.add(r_key)
                
            if s_key in seen_section: 
                conflicts += 200; bad_indices.add(i)
            else: seen_section.add(s_key)

            if d_key in seen_daily_class:
                conflicts += 10; bad_indices.add(i)
            else:
                seen_daily_class.add(d_key)
                
        return conflicts, list(bad_indices)

    def optimize(self, time_limit_seconds=15.0):
        # We will loop aggressively until we hit 0 or run out of our safe time limit.
        start_time = time.time()
        restarts = 0
        
        while self.best_fitness > 0 and (time.time() - start_time) < time_limit_seconds:
            stagnation_counter = 0
            prev_best_fitness = self.best_fitness
            
            for iteration in range(self.max_iter):
                for i in range(self.pop_size):
                    fit, bad_indices = self.calculate_fitness(self.population[i])
                    if fit < self.best_fitness:
                        self.best_fitness = fit
                        self.best_schedule = np.copy(self.population[i])
                        self.best_bad_indices = bad_indices
                
                # Check for stagnation
                if self.best_fitness < prev_best_fitness:
                    stagnation_counter = 0
                    prev_best_fitness = self.best_fitness
                else:
                    stagnation_counter += 1
                
                if self.best_fitness == 0 or (time.time() - start_time) > time_limit_seconds: 
                    break
                
                # If we haven't improved in 30 generations, break out to trigger a Random Restart
                if stagnation_counter > 30:
                    break
                
                # First-Choice Hill Climbing
                if self.best_fitness > 0 and len(self.best_bad_indices) > 0:
                    discrete_slots = np.clip(np.floor(self.best_schedule * self.total_slots).astype(int), 0, self.total_slots - 1)
                    random.shuffle(self.best_bad_indices)
                    target_indices = self.best_bad_indices[:40] 
                    
                    for c_idx in target_indices:
                        original_slot = discrete_slots[c_idx]
                        test_slots = random.sample(range(self.total_slots), min(15, self.total_slots))
                        
                        for temp_slot in test_slots:
                            if temp_slot == original_slot: continue
                            
                            discrete_slots[c_idx] = temp_slot
                            temp_continuous = (discrete_slots + 0.5) / self.total_slots
                            temp_fit, _ = self.calculate_fitness(temp_continuous)
                            
                            if temp_fit < self.best_fitness:
                                self.best_fitness = temp_fit
                                self.best_schedule = temp_continuous
                                break 
                            else:
                                discrete_slots[c_idx] = original_slot
                                
                        if self.best_fitness == 0 or (time.time() - start_time) > time_limit_seconds: 
                            break
                
                if self.best_fitness == 0 or (time.time() - start_time) > time_limit_seconds: 
                    break

                new_population = []
                for _ in range(self.pop_size // 2):
                    p1_idx, p2_idx = random.randint(0, self.pop_size-1), random.randint(0, self.pop_size-1)
                    cpt = random.randint(1, self.num_classes - 1)
                    child = np.concatenate((self.population[p1_idx][:cpt], self.population[p2_idx][cpt:]))
                    if random.random() < 0.1: child[random.randint(0, self.num_classes-1)] = random.random()
                    new_population.append(child)
                    
                h = 5 - (iteration * (5 / self.max_iter))
                for i in range(self.pop_size // 2):
                    sho_child = np.copy(self.population[i])
                    for j in range(self.num_classes):
                        B, E = 2 * np.random.random(), 2 * h * np.random.random() - h
                        sho_child[j] = np.clip(self.best_schedule[j] - E * abs(B * self.best_schedule[j] - sho_child[j]), 0, 0.999)
                    new_population.append(sho_child)
                self.population = np.array(new_population)

            # --- THE RANDOM RESTART MECHANISM ---
            # If we broke out of the generation loop but still aren't at 0, we nuke the population 
            # and start fresh, but keep the current best schedule in slot 0.
            if self.best_fitness > 0 and (time.time() - start_time) < time_limit_seconds:
                restarts += 1
                self.population = np.random.rand(self.pop_size, self.num_classes)
                if self.best_schedule is not None:
                    self.population[0] = self.best_schedule
                    
        print(f"    [Info] Algorithm executed {restarts} strategic random restarts.")
                    
        slots = np.floor(self.best_schedule * self.total_slots).astype(int)
        timetable = []
        for i, c in enumerate(self.courses):
            timetable.append({
                "course_code": c.instance_code, 
                "course_name": c.course_name,
                "assigned_room": self.course_room_map.get(c.instance_code),
                "assigned_teacher": self.fast_eval_data[i][0], 
                "time_slot": int(slots[i]), "year": c.target_section.year, "section": c.target_section.section
            })
        return timetable, self.best_fitness

class UserAuth(BaseModel): username: str; password: str; role: str; year: Optional[int] = None; section: Optional[str] = None
class GenConfig(BaseModel): days_per_week: int = 5; slots_per_day: int = 8
class StudentUpdate(BaseModel): year: int; section: str
class CourseAssign(BaseModel): year: int; section: str

class SubRequestPayload(BaseModel):
    requester_username: str
    target_teacher_username: str
    time_slot: int
    course_code: str
    year: int
    section: str

@app.post("/api/auth/register")
def register(user: UserAuth, db: Session = Depends(get_db)):
    if db.query(DBUser).filter(DBUser.username == user.username).first(): raise HTTPException(status_code=400)
    hashed_pwd = pwd_context.hash(user.password)
    new_user = DBUser(username=user.username, password_hash=hashed_pwd, role=user.role, year=user.year, section=user.section)
    if user.role == "teacher":
        new_t = DBTeacher(name=user.username, expertise=[])
        db.add(new_t); db.commit(); db.refresh(new_t)
        new_user.teacher_id = new_t.id
    db.add(new_user); db.commit()
    return {"status": "User created"}

@app.post("/api/auth/login")
def login(user: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password_hash): raise HTTPException(status_code=401)
    token = jwt.encode({"sub": db_user.username, "role": db_user.role, "id": db_user.id}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "role": db_user.role, "username": db_user.username, "year": db_user.year, "section": db_user.section, "teacher_id": db_user.teacher_id}

@app.get("/api/courses")
def get_all_courses(db: Session = Depends(get_db)):
    return [{"course_code": c.course_code, "course_name": c.course_name} for c in db.query(DBCourse).all()]

@app.post("/api/generate")
def generate_schedule(config: GenConfig, db: Session = Depends(get_db)):
    print("\n" + "="*60)
    print("🕒 TIMETABLE GENERATION PROCESS STARTED")
    print("="*60)
    total_start_time = time.time()
    
    db_courses = db.query(DBCourse).all()
    teachers = db.query(DBTeacher).all()
    rooms = db.query(DBRoom).all()

    class class_instance: pass
    instances = []
    
    for c in db_courses:
        for sec in c.target_sections:
            for _ in range(c.lectures_per_week):
                inst = class_instance()
                inst.base_code = c.course_code
                inst.instance_code = f"{c.course_code}_LEC"
                inst.course_name = c.course_name
                inst.target_section = sec
                instances.append(inst)
            for _ in range(c.labs_per_week):
                inst = class_instance()
                inst.base_code = c.course_code
                inst.instance_code = f"{c.course_code}_LAB"
                inst.course_name = f"{c.course_name} (Lab)"
                inst.target_section = sec
                instances.append(inst)

    if not instances: raise HTTPException(status_code=400, detail="No courses assigned.")

    print("--> 1. Running AI Room Mapping (Gemini)...")
    room_start = time.time()
    course_room_map = ai_enrich_and_map_rooms(db_courses, rooms)
    room_end = time.time()
    print(f"    [Done] AI Room Mapping took: {round(room_end - room_start, 4)} seconds")

    # Time limit set to 15 seconds safely
    solver = HybridGeneticSHOSolver(instances, teachers, course_room_map, config.days_per_week * config.slots_per_day, config.days_per_week)
    print("--> 2. Running Hybrid Genetic SHO Algorithm (Targeting Penalty: 0)...")
    solver_start = time.time()
    timetable, fitness = solver.optimize(time_limit_seconds=15.0) 
    solver_end = time.time()
    print(f"    [Done] SHO Algorithm took: {round(solver_end - solver_start, 4)} seconds")
    
    total_end_time = time.time()
    execution_time_sec = round(total_end_time - total_start_time, 4)
    
    print("-" * 60)
    print(f"✅ TOTAL GENERATION TIME: {execution_time_sec} seconds")
    
    print("=" * 60 + "\n")
    
    db.add(DBSchedule(schedule_data=timetable)); db.commit()
    return {"status": "success", "penalty": fitness, "execution_time_seconds": execution_time_sec, "timetable": timetable}

@app.get("/api/schedule")
def get_schedule(db: Session = Depends(get_db)):
    sched = db.query(DBSchedule).order_by(DBSchedule.id.desc()).first()
    return {"timetable": sched.schedule_data if sched else []}

@app.get("/api/admin/dashboard")
def get_admin_dashboard(db: Session = Depends(get_db)):
    sections = db.query(DBYearSection).all()
    res_sections = []
    for sec in sections:
        students = db.query(DBUser).filter(DBUser.role == "student", DBUser.year == sec.year, DBUser.section == sec.section).all()
        courses = [{"id": c.id, "code": c.course_code, "name": c.course_name, "L": c.lectures_per_week, "P": c.labs_per_week} for c in sec.courses]
        res_sections.append({
            "id": sec.id, "year": sec.year, "section": sec.section,
            "enrolled_students": len(students), "student_names": [s.username for s in students], "assigned_courses": courses
        })
    
    res_courses = []
    for c in db.query(DBCourse).all():
        res_courses.append({
            "id": c.id, "course_code": c.course_code, "course_name": c.course_name,
            "L": c.lectures_per_week, "P": c.labs_per_week,
            "assigned_to": [f"Yr {s.year}-{s.section}" for s in c.target_sections], 
            "assigned_section_ids": [s.id for s in c.target_sections]
        })
    return {"sections": res_sections, "all_courses": res_courses}

@app.put("/api/admin/student/{student_username}")
def update_student(student_username: str, data: StudentUpdate, db: Session = Depends(get_db)):
    student = db.query(DBUser).filter(DBUser.username == student_username).first()
    student.year = data.year; student.section = data.section
    db.commit()
    return {"status": "success"}

@app.put("/api/admin/course/{course_id}/assign")
def assign_course(course_id: int, data: CourseAssign, db: Session = Depends(get_db)):
    course = db.query(DBCourse).filter(DBCourse.id == course_id).first()
    sec = db.query(DBYearSection).filter(DBYearSection.year == data.year, DBYearSection.section == data.section).first()
    if not sec:
        sec = DBYearSection(year=data.year, section=data.section, student_count=0); db.add(sec); db.commit(); db.refresh(sec)
    if sec not in course.target_sections: course.target_sections.append(sec); db.commit()
    return {"status": "success"}

@app.put("/api/admin/course/{course_id}/unassign")
def unassign_course(course_id: int, data: CourseAssign, db: Session = Depends(get_db)):
    course = db.query(DBCourse).filter(DBCourse.id == course_id).first()
    sec = db.query(DBYearSection).filter(DBYearSection.year == data.year, DBYearSection.section == data.section).first()
    if sec in course.target_sections: course.target_sections.remove(sec); db.commit()
    return {"status": "success"}

# --- SUBSTITUTION SYSTEM ENDPOINTS WITH AI INFERENCE ---

@app.get("/api/teachers/free/{time_slot}")
def get_free_teachers(time_slot: int, course_code: str, db: Session = Depends(get_db)):
    """Finds free teachers and uses AI to infer their expertise based on their current schedule."""
    print("\n" + "="*60)
    print("🕒 AI SUBSTITUTION INFERENCE STARTED")
    print("="*60)
    total_start_time = time.time()
    
    sched = db.query(DBSchedule).order_by(DBSchedule.id.desc()).first()
    
    if not sched or not sched.schedule_data:
        return {"teachers": []}

    busy_teachers = set()
    teacher_portfolios = {} 

    # 1. Build teacher portfolios
    for item in sched.schedule_data:
        t_name = item.get("assigned_teacher")
        c_code = item.get("course_code").replace("_LEC", "").replace("_LAB", "")
        
        if item.get("time_slot") == time_slot:
            busy_teachers.add(t_name)
            
        if t_name not in teacher_portfolios:
            teacher_portfolios[t_name] = set()
        teacher_portfolios[t_name].add(c_code)

    all_teachers = db.query(DBTeacher).all()
    free_teachers = [t.name for t in all_teachers if t.name not in busy_teachers]

    if not free_teachers:
        return {"teachers": []}

    # 2. Extract target course info
    base_course_code = course_code.replace("_LEC", "").replace("_LAB", "")
    target_course = db.query(DBCourse).filter(DBCourse.course_code == base_course_code).first()
    target_course_name = target_course.course_name if target_course else base_course_code

    # 3. Format data for Gemini
    free_teachers_data = {}
    for t in free_teachers:
        t_courses = teacher_portfolios.get(t, set())
        course_details = []
        for cc in t_courses:
            c = db.query(DBCourse).filter(DBCourse.course_code == cc).first()
            if c: course_details.append(f"{cc} ({c.course_name})")
            else: course_details.append(cc)
        free_teachers_data[t] = course_details

    # 4. AI Expertise Matching Timer
    print("--> Running AI Expertise Matching (Gemini)...")
    ai_start = time.time()
    
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = f"""
        Target Course needing a substitute: {base_course_code} ({target_course_name})
        
        Available Teachers and the courses they are currently assigned to teach:
        {json.dumps(free_teachers_data, indent=2)}
        
        Task: Identify which of these available teachers have the right academic expertise to teach the Target Course. 
        Infer their expertise entirely from the subjects they currently teach. If they teach a subject in the same domain 
        (e.g., if target is English, find teachers teaching other English/Humanities courses. If target is CS, find CS teachers).
        
        Return ONLY a JSON array of strings containing the exact names of the matching teachers. Do not include markdown formatting.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        capable_teachers = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        
        ai_end = time.time()
        print(f"    [Done] AI Inference took: {round(ai_end - ai_start, 4)} seconds")
        
        capable_teachers = [t for t in capable_teachers if t in free_teachers]
        
        total_end_time = time.time()
        print("-" * 60)
        print(f"✅ TOTAL SUBSTITUTION INFERENCE TIME: {round(total_end_time - total_start_time, 4)} seconds")
        print("=" * 60 + "\n")
        
        return {"teachers": capable_teachers}
        
    except Exception as e:
        ai_end = time.time()
        print(f"    [Failed] AI Inference errored after: {round(ai_end - ai_start, 4)} seconds - {e}")
        # Fallback to simple matching if AI fails
        fallback_matches = [t for t in free_teachers if any(c.startswith(base_course_code[:3]) for c in teacher_portfolios.get(t, set()))]
        return {"teachers": fallback_matches}

@app.post("/api/substitution/request")
def create_substitution_request(payload: SubRequestPayload, db: Session = Depends(get_db)):
    new_request = DBSubstitutionRequest(
        requester_username=payload.requester_username,
        target_teacher_username=payload.target_teacher_username,
        time_slot=payload.time_slot,
        course_code=payload.course_code,
        year=payload.year,
        section=payload.section,
        status="pending"
    )
    db.add(new_request)
    db.commit()
    return {"status": "success"}

@app.get("/api/substitution/incoming/{username}")
def get_incoming_requests(username: str, db: Session = Depends(get_db)):
    requests = db.query(DBSubstitutionRequest).filter(
        DBSubstitutionRequest.target_teacher_username == username,
        DBSubstitutionRequest.status == "pending"
    ).all()
    return requests

@app.post("/api/substitution/accept/{request_id}")
def accept_substitution_request(request_id: int, db: Session = Depends(get_db)):
    req = db.query(DBSubstitutionRequest).filter(DBSubstitutionRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
        
    req.status = "accepted"
    sched = db.query(DBSchedule).order_by(DBSchedule.id.desc()).first()
    
    if sched and sched.schedule_data:
        updated_schedule = [dict(item) for item in sched.schedule_data]
        for item in updated_schedule:
            if (item.get("time_slot") == req.time_slot and 
                item.get("course_code") == req.course_code and 
                item.get("year") == req.year and 
                item.get("section") == req.section):
                
                item["assigned_teacher"] = req.target_teacher_username
                
        db.query(DBSchedule).filter(DBSchedule.id == sched.id).update({"schedule_data": updated_schedule})
        
    db.commit()
    return {"status": "success"}