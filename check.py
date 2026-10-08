sessions = [
    {"subject": "A", "minutes": 4},
    {"subject": "A", "minutes": 6}
]
totals = {"A": sessions[0]["minutes"] + sessions[1]["minutes"]}
sessions[0]["minutes"] = 9
print(totals["A"])
print(sessions[0]["minutes"] + sessions[1]["minutes"])
