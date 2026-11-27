import os
import tempfile

import dashboard

dashboard.LOG_PATH = os.path.join(tempfile.mkdtemp(), "mood_log.csv")  # never touch the real log

for t in range(1000, 1032):  # 31 seconds of Sad, one frame per second
    dashboard.update("Sad", 0.5, now=t)
state = dashboard.snapshot(now=1031.5)
assert state["mood"] == "Sad" and state["check_in"]
assert not dashboard.snapshot(now=1040)["check_in"]  # face gone: no banner
assert dashboard.snapshot(now=1040)["mood"] is None

dashboard.update("Sad", 0.5, now=1100)  # face comes back: timer restarts, no new CSV row
assert dashboard.snapshot(now=1100)["seconds"] == 0
dashboard.update("Happy", 0.8, now=1101)
assert dashboard.snapshot(now=1101)["share"] == {"Sad": 32, "Happy": 0}
assert dashboard.snapshot(now=1101 + 601)["timeline"] == []  # older than 10 minutes

with open(dashboard.LOG_PATH) as f:
    rows = [line.strip().split(",")[1:] for line in f]
assert rows == [["emotion", "score"], ["Sad", "0.50"], ["Happy", "0.80"]]

client = dashboard.app.test_client()
assert client.get("/").status_code == 200
assert "mood" in client.get("/api/state").get_json()
print("all dashboard checks OK")
