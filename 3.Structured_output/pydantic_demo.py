from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int
    grade: str
    
student_data = {
    "name": "Alice",
    "age": 20,
    "grade": "A"
}

student = Student(**student_data)
print(student)