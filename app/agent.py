def check_available_slots():

    return {
        "department": "Cardiology",
        "available_slots": [
            "10:00 AM",
            "12:00 PM",
            "03:00 PM"
        ]
    }


def route_question(question: str):

    appointment_keywords = [
        "appointment",
        "book",
        "schedule",
        "slot"
    ]

    question = question.lower()

    for keyword in appointment_keywords:
        if keyword in question:
            return {
                "type": "tool",
                "response": check_available_slots()
            }

    return {
        "type": "rag"
    }